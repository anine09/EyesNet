import os
import gymnasium as gym
import numpy as np
from gymnasium.spaces import MultiDiscrete, Box
from stable_baselines3 import PPO

# from sbx import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from loguru import logger
import requests
from sklearn.decomposition import PCA
import time
import subprocess
from python_retry import retry
import shutil
from icecream import ic

# DATA_DIR = "C:/Users/Epsilon_Luoo/Downloads/All_files/real_xrd"
# GPX_PATH = "Ti3(BiO3)4.gpx"
# CIF_PATH = DATA_DIR + "/Ti3(BiO3)4_1.cif"
# POWER_DATA_PATH = DATA_DIR + "/xrd_3.csv"
# INST_PARAMS_PATH = DATA_DIR + "/../INST_XRY.PRM"

DATA_DIR = "C:/Users/Epsilon_Luoo/Downloads/All_files"
GPX_PATH = "fap.gpx"
CIF_PATH = DATA_DIR + "/FAP.EXP"
POWER_DATA_PATH = DATA_DIR + "/FAP.XRA"
INST_PARAMS_PATH = DATA_DIR + "/INST_XRY.PRM"


class GSASWorld(gym.Env):
    metadata = {"render.modes": ["ascii"]}
    action_list = [
        "background_function",
        "instrument_U",
        "instrument_V",
        "instrument_W",
        "instrument_X",
        "instrument_Y",
        "instrument_Z",
        # "instrument_alpha", # error
        # "instrument_beta-0", # error
        # "instrument_beta-1", # error
        # "instrument_beta-q", # error
        # "instrument_sig-0", # error
        # "instrument_sig-1", # error
        # "instrument_sig-2", # error
        # "instrument_sig-q", # error
        # "instrument_difA", # error
        # "instrument_difB", # error
        # "instrument_difC", # error
        "instrument_zero",
        "instrument_SH/L",
        "instrument_Polariz.",
        # "instrument_Lam", # error
        # "sigle_xtral_scale",
        # "sigle_xtral_BabA",
        # "sigle_xtral_BabU",
        # "sigle_xtral_Eg",
        # "sigle_xtral_Es",
        # "sigle_xtral_Ep",
        # "sigle_xtral_Flack",
        "sample_displacement_y",
        # "sample_displacement_x",
        # "sample_absorption",
        # "sample_contrast",
        # "sample_scale",
        # "unit_cell",
        "LeBail",
        "cell_size",
        "cell_microstrain",
        # "HStrain",
        "atom_X",
        "atom_U",
        "atom_F",
        # "Babinet_BabA",
        # "Babinet_BabU",
        # "Extinction",
        "Pref.Ori.",
        # "Show",
        # "Use",
        # "Scale",
    ]
    action_map = {idx: action for idx, action in enumerate(action_list)}

    def __init__(
        self,
        gsas_url,
        gpx_path,
        powder_data_path,
        inst_parmas_path,
        cif_path,
        data_dir,
        operation_num=1,
        background_fit_times=10,
        wR_threshold=10,
        delta_threshold=1e-4,
        feature_num=200,
    ):
        self.gsas_url = gsas_url
        self.gpx_path = gpx_path
        self.powder_data_path = powder_data_path
        self.inst_parmas_path = inst_parmas_path
        self.cif_path = cif_path
        self.data_dir = data_dir
        self.files_path = {
            "gpx_path": self.gpx_path,
            "powder_data_path": self.powder_data_path,
            "inst_parmas_path": self.inst_parmas_path,
            "cif_path": self.cif_path,
            "data_dir": self.data_dir,
        }

        self.operation_num = operation_num
        self.background_fit_times = background_fit_times
        self.wR_threshold = wR_threshold
        self.delta_threshold = delta_threshold
        self.feature_num = feature_num

        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(self.feature_num,), dtype=np.float64
        )
        self.action_space = MultiDiscrete([len(self.action_list)] * operation_num)

        self.action = None
        self.pac = PCA(n_components=1)

        self.truncation_times = 0

    def _pac(self, data):
        features = self.pac.fit_transform(data)
        return features

    def _get_residuel_features(self):
        residual = self.residual.copy()
        drop_len = len(residual) % self.feature_num
        residual_dropped = residual[:-drop_len].reshape(self.feature_num, -1)
        residual_features = self._pac(residual_dropped)
        return residual_features.flatten()

    def step(self, action):
        self.actions = [self.action_map[act] for act in action]
        refine_data = {
            "proj_id": self.proj_id,
            "actions": self.actions,
        }
        refine_data = requests.post(
            self.gsas_url + "/do_refinements", json=refine_data
        ).json()
        self.residual = np.array(refine_data["residual"])
        new_wR = refine_data["wR"]
        truncation = refine_data["truncation"]
        if new_wR > self.init_wR + 5 or new_wR == 100:
            truncation = True

        reward = self.wR - new_wR
        self.total_reward += reward
        observation = self._get_residuel_features()
        self.wR = new_wR

        if self.wR < self.wR_threshold and reward < self.delta_threshold:
            terminated = True
            requests.get(
                self.gsas_url + "/save_project", params={"proj_id": self.proj_id}
            )
            logger.info(f"Project {self.proj_id} saved")
            self.terminated = True
        else:
            terminated = False
            self.terminated = False

        logger.info(
            f"action: {self.actions}\twR: {round(self.wR, 4)}\treward: {round(reward, 4)}"
        )
        with open(f"logs/info_{self.proj_id}.csv", "a") as f:
            f.write(f"{self.actions},{self.wR},{reward},{self.total_reward}\n")

        if truncation:
            self.truncation_times += 1
            with open(f"logs/truncation.log", "a") as f:
                f.write(
                    f"{time.ctime()}:\t{self.actions},\t{self.wR},\t{reward}\t->{self.truncation_times}: {self.proj_id}\n"
                )
            logger.error("Truncated")
            if self.truncation_times >= 10:
                exit()
        else:
            self.truncation_times = 0

        if terminated:
            logger.info("Terminated")

        assert self.truncation_times < 10, "Too many truncations."

        return observation, reward, terminated, truncation, {}

    def reset(self, seed=None):
        super().reset(seed=seed)
        if hasattr(self, "proj_id") and not self.terminated:
            rsp = requests.get(
                self.gsas_url + "/remove_project", params={"proj_id": self.proj_id}
            )
            logger.info(rsp.text)
            self.terminated = False
        rsp = requests.post(self.gsas_url + "/create_project", json=self.files_path)
        self.proj_id = rsp.json()["proj_id"]
        logger.add(f"logs/{self.proj_id}.log")
        logger.info(rsp.text)
        with open(f"logs/info_{self.proj_id}.csv", "w") as f:
            f.write("action,wR,reward,total_reward\n")

        fit_info = requests.post(
            self.gsas_url + "/fit_background",
            params={"proj_id": self.proj_id, "study_epoch": self.background_fit_times},
        ).json()
        logger.info(fit_info)
        residual_data = requests.get(
            self.gsas_url + "/get_residual", params={"proj_id": self.proj_id}
        ).json()
        self.residual = np.array(residual_data["residual"])
        self.wR = fit_info["wR"]
        self.init_wR = fit_info["wR"]
        self.total_reward = 0
        observation = self._get_residuel_features()
        return observation, {}


@retry(max_retries=10)
def main():
    print("killing uvicorn.exe")
    os.system("taskkill /f /im uvicorn.exe")
    time.sleep(5)
    print("starting uvicorn.exe")
    subprocess.Popen("uvicorn gsas_api:app", shell=True)
    time.sleep(5)

    gsas_env = GSASWorld(
            gsas_url="http://localhost:8000",
            gpx_path=GPX_PATH,
            powder_data_path=POWER_DATA_PATH,
            inst_parmas_path=INST_PARAMS_PATH,
            cif_path=CIF_PATH,
            data_dir=DATA_DIR,
            operation_num=1,
            background_fit_times=20,
        )


    model = PPO("MlpPolicy", gsas_env, verbose=1)
    model.learn(total_timesteps=1e4)


if __name__ == "__main__":
    main()
