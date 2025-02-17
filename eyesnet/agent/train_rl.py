# from stable_baselines3 import PPO
# from sbx import PPO
from sb3_contrib import RecurrentPPO
import time
from GSASEnv import GSASEnv
import swanlab


def main():
    DATASET_PATH = "/home/lxt/EyesNet/dataset/crystal_graph/XRD_exp"
    OUTPUT_PATH = "/home/lxt/EyesNet/eyesnet/agent/rl_train"

    env = GSASEnv(
        "http://localhost:8000",
        DATASET_PATH,
        OUTPUT_PATH,
        background_fit_times=15,
        operation_num=1,

    )
    time.sleep(1)

    model = RecurrentPPO("MlpLstmPolicy", env)
    model.learn(total_timesteps=1024, progress_bar=True)
    swanlab.finish()
    model.save("rl_train/GSAS_PPO_model")


if __name__ == "__main__":
    main()
