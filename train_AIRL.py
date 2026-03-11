import pickle
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.ppo import MlpPolicy

from imitation.algorithms.adversarial.airl import AIRL
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.policies.serialize import load_policy
from imitation.rewards.reward_nets import BasicShapedRewardNet
from imitation.util.networks import RunningNorm
from imitation.util.util import make_vec_env
from imitation.util import logger as imit_logger


import os

# 导入你的自定义环境类
from Aircombat_env_for_AIRL import FlightEnv

if __name__ == "__main__":
    # --- 1. 设置保存路径 ---
    log_path = "./airl_results"
    os.makedirs(log_path, exist_ok=True)

    # --- 2. 关键：配置全局 Logger 输出为 CSV ---
    # 这会捕获 airl_trainer 所有的控制台打印信息
    custom_logger = imit_logger.configure(
        folder=log_path,
        format_strs=["stdout", "csv", "tensorboard"]
    )

    SEED = 42
    gym.register(
        id="custom/ObservationMatching-v0",
        entry_point=FlightEnv,
        # This can also be the path to the class, e.g. `observation_matching:ObservationMatchingEnv`
    )

    # Create a single environment for training an expert with SB3
    env = gym.make("custom/ObservationMatching-v0")

    # Create a vectorized environment for training with `imitation`

    # Option A: use the `make_vec_env` helper function - make sure to pass `post_wrappers=[lambda env, _: RolloutInfoWrapper(env)]`
    venv = make_vec_env(
        "custom/ObservationMatching-v0",
        rng=np.random.default_rng(SEED),
        n_envs=2,
        post_wrappers=[lambda env, _: RolloutInfoWrapper(env)],
    )

    # --- A. 加载离线生成的归一化轨迹 ---
    print("Loading preprocessed expert trajectories...")
    with open("expert_traj_norm.pkl", 'rb') as f:
        expert_trajs = pickle.load(f)
    print(f"Loaded {len(expert_trajs)} normalized trajectories.")

    # --- B. 加载统计量用于环境同步 ---
    stats = np.load("obs_stat.npz")
    obs_mean = stats['mean']
    obs_std = stats['std']

    learner = PPO(
        env=venv,
        policy=MlpPolicy,
        n_steps=4096,
        batch_size=2048,
        ent_coef=0.001,
        learning_rate=0.0005,
        gamma=0.95,
        clip_range=0.1,
        vf_coef=0.1,
        n_epochs=10,
        tensorboard_log=log_path,  # 告知 SB3 基础路径
        seed=SEED,
        verbose=1,
    )
    reward_net = BasicShapedRewardNet(
        observation_space=env.observation_space,
        action_space=env.action_space,
        normalize_input_layer=RunningNorm,
    )
    learner.set_logger(custom_logger)
    airl_trainer = AIRL(
        demonstrations=expert_trajs,
        demo_batch_size=512,
        gen_replay_buffer_capacity=4096,
        n_disc_updates_per_round=5,
        venv=venv,
        gen_algo=learner,
        reward_net=reward_net,
        allow_variable_horizon=True,
    )
    airl_trainer._logger = custom_logger

    airl_trainer.train(1000000)  # Train for 2_000_000 steps to match expert.
    custom_logger.close()