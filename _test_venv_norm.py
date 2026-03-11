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
from stable_baselines3.common.vec_env import VecNormalize

import os

# 导入你的自定义环境类
from Aircombat_env_for_AIRL import FlightEnv

if __name__ == "__main__":


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
        n_envs=1,
        post_wrappers=[lambda env, _: RolloutInfoWrapper(env)],
    )

    venv = VecNormalize(venv, training=True, norm_obs=True, norm_reward=False)

    # --- 2. 交互循环 ---
    obs = venv.reset()  # 返回的 obs 已经是归一化后的，形状为 (2, 12)

    for _ in range(1000):
        # PPO 预测时也需要传入归一化后的 obs
        single_action = np.array([0, 0, 0, 0.8])
        action = np.expand_dims(single_action, axis=0)  # 变为 (1, 4)
        # 执行动作
        # venv.step 内部会自动将 action 分发给 2 个子环境
        # 返回的 next_obs 也是自动归一化后的
        obs, rewards, dones, infos = venv.step(action)
        print(obs)
        if dones:
            break