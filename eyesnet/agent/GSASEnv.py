import json
import os
import subprocess
import sys
import time

import requests
from icecream import ic
from stable_baselines3.common.env_checker import check_env

sys.path.insert(0, "/home/lxt/EyesNet/eyesnet")
import random
import tomllib

import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces import Box, MultiDiscrete
from reranker.config_model import Config
from reranker.eyesnet_clip import EyesNetCLIP

action_list = [
    "background_function",
    "instrument_U",
    "instrument_V",
    "instrument_W",
    "instrument_X",
    "instrument_Y",
    "instrument_Z",
    "instrument_zero",
    "instrument_SH/L",
    "instrument_Polariz.",
    "unit_cell",
    "cell_size",
    "cell_microstrain",
    "atom_X",
    "atom_U",
    "atom_F",
]

action_map = {idx: action for idx, action in enumerate(action_list)}

MODEL_CONFIG_PATH = "/home/lxt/EyesNet/eyesnet/reranker/config.toml"
CHECKPOINT_PATH = "/home/lxt/EyesNet/eyesnet/reranker/latest_step_model.pth"

with open(MODEL_CONFIG_PATH, "rb") as f:
    config = tomllib.load(f)
config = Config(**config)

INST_PARAMS_PATH = "/home/lxt/EyesNet/eyesnet/agent/CuKa_lab_data.instprm"


def xrd2tensor(xrd_data):
    xrd_data = torch.tensor(xrd_data)
    xrd_data = torch.abs(torch.round(xrd_data))
    min_val = xrd_data.min()
    max_val = xrd_data.max()
    xrd_data = (xrd_data - min_val) / (max_val - min_val) * 10000
    xrd_data = xrd_data.int()
    return xrd_data


def background_decay(x, A=1.0, k=2.0):
    return A * 1.5 ** (-k * x)


def make_gsas_input_file(file_name, output_dir):
    with open(file_name, "r") as f:
        data = json.load(f)
    with open(f"{output_dir}/gen_data.cif", "w") as f:
        f.write(data["CIF_info"])
    xrd_seq = random.sample(data["random_XRD_simulation"], 1)
    xrd_seq = np.array(xrd_seq)
    background_simulation = background_decay(
        np.linspace(10, 80, 3501), A=xrd_seq.max() * 0.2, k=1 / 3
    )
    xrd_seq = xrd_seq + background_simulation
    xrd_seq = xrd2tensor(xrd_seq).tolist()[0]
    with open(f"{output_dir}/xrd_seq.csv", "w") as f:
        f.write("x,y_obs\n")
        for x, y in zip(range(1000, 8000 + 2, 2), xrd_seq):
            f.write(f"{x/100},{y}\n")


class GSASEnv(gym.Env):
    def __init__(
        self,
        gsas_url,
        input_dir,
        output_dir,
        feature_num=128,
        operation_num=1,
        background_fit_times=30,
        wR_threshold=15,
        delta_threshold=1e-2,
        wR_patience=5,
        max_operation_step=100,
    ):
        self.gsas_url = gsas_url
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.background_fit_times = background_fit_times
        self.wR_threshold = wR_threshold
        self.delta_threshold = delta_threshold
        self.wR_patience = wR_patience
        self.delta_wR_times = 0
        self.max_operation_step = max_operation_step
        self.operation_step = 0

        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(2, feature_num), dtype=np.float64
        )
        self.action_space = MultiDiscrete([len(action_list)] * operation_num)

        model = EyesNetCLIP(config)
        checkpoint = torch.load(CHECKPOINT_PATH, weights_only=True, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        self.embedding_model = model.xrd_encoder

        self.file_list = os.listdir(self.input_dir)
        self.file_list = random.sample(self.file_list, 2048)
        
        self.file_list = iter(self.file_list)

        self.refine_status = True

    def _embedding(self, data):
        feature = self.embedding_model(data)
        return feature

    def _get_observation(self):
        current_hist = xrd2tensor([self.Yobs, self.Ycalc])
        feature = self._embedding(current_hist)
        feature = feature.detach().cpu().numpy()
        return feature

    def step(self, action):
        self.actions = [action_map[act] for act in action]
        refine_actions = {
            "proj_id": self.proj_id,
            "actions": self.actions,
        }
        try:
            refine_data = requests.post(
                self.gsas_url + "/do_refinements", json=refine_actions
            ).json()
        except Exception as e:
            print(e)
            reward = 0
            truncation = True


        self.Ycalc = refine_data["Ycalc"]
        new_wR = refine_data["wR"]
        is_success = refine_data["is_success"]

        reward = self.wR - new_wR
        self.wR = new_wR

        if not is_success:
            reward = -1000
            terminated = True
        elif new_wR == 100:
            reward = -1000
            terminated = True
        elif new_wR < self.wR_threshold and reward < self.delta_threshold:
            if self.delta_wR_times >= self.wR_patience:
                reward = 1000
                terminated = True
                save_msg = requests.post(
                    self.gsas_url + "/save_project", params={"proj_id": self.proj_id}
                ).text
                ic(save_msg)
                self.delta_wR_times = 0
            else:
                self.delta_wR_times += 1
                terminated = False
        else:
            terminated = False

        if self.operation_step >= self.max_operation_step:
            truncation = True
        else:
            truncation = False
            self.operation_step += 1

        observation = self._get_observation()

        self.refine_status = is_success
        print("=" * 60)
        print(
            f"wR: {new_wR}\nActions: {set(self.actions)}\nReward: {reward}\nStep: {self.operation_step}"
        )
        print("=" * 60)
        if (reward == -1000) or truncation:
            rm_msg = requests.get(
                self.gsas_url + "/remove_project", params={"proj_id": self.proj_id}
            ).text
            ic(rm_msg)
        return observation, reward, terminated, truncation, {}

    def reset(self, seed=None):
        super().reset(seed=seed)

        print("killing uvicorn")
        os.system("pkill -9 -f 'uvicorn new_gsas_api:app'")
        time.sleep(1)
        print("starting uvicorn.exe")
        subprocess.Popen("uvicorn new_gsas_api:app", shell=True)
        time.sleep(5)

        self.current_file = next(self.file_list)
        self.operation_step = 0

        file_name = os.path.join(self.input_dir, self.current_file)
        make_gsas_input_file(file_name, self.output_dir)
        file_data = {
            "powder_data_path": self.output_dir + "/xrd_seq.csv",
            "inst_parmas_path": INST_PARAMS_PATH,
            "cif_path": self.output_dir + "/gen_data.cif",
            "output_dir": self.output_dir,
        }

        rsp = requests.post(self.gsas_url + "/create_project", json=file_data)
        self.proj_id = rsp.text.replace('"', "")

        background_fit_info = requests.post(
            self.gsas_url + "/fit_background",
            params={"proj_id": self.proj_id, "study_epoch": self.background_fit_times},
        ).json()
        ic(background_fit_info)
        self.wR = background_fit_info["wR"]

        self.Yobs = requests.get(
            self.gsas_url + "/get_Yobs", params={"proj_id": self.proj_id}
        ).json()["Yobs"]

        self.Ycalc = requests.get(
            self.gsas_url + "/get_Ycalc", params={"proj_id": self.proj_id}
        ).json()["Ycalc"]

        observation = self._get_observation()

        return observation, {}


if __name__ == "__main__":
    DATASET_PATH = "/home/lxt/EyesNet/dataset/crystal_graph/XRD_exp"
    OUTPUT_PATH = "/home/lxt/EyesNet/eyesnet/agent/rl_train"

    print("killing uvicorn")
    os.system("pkill -9 -f 'uvicorn new_gsas_api:app'")
    time.sleep(1)
    print("starting uvicorn.exe")
    subprocess.Popen("uvicorn new_gsas_api:app", shell=True)
    time.sleep(1)

    env = GSASEnv(
        "http://localhost:8000",
        DATASET_PATH,
        OUTPUT_PATH,
        background_fit_times=1,
    )
    time.sleep(1)
    check_env(env)
