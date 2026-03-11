import numpy as np
import pickle
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from imitation.algorithms.adversarial.airl import AIRL
from imitation.rewards.reward_nets import BasicShapedRewardNet
from imitation.util.networks import RunningNorm
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from imitation.data.wrappers import RolloutInfoWrapper

def make_env():
    env = FlightEnv(norm_mean=obs_mean, norm_std=obs_std)
    env = RolloutInfoWrapper(env) # 必须有这个，AIRL 才能获取轨迹信息
    return env


# 导入你的自定义环境类
from Aircombat_env_for_AIRL import FlightEnv

if __name__ == "__main__":
    SEED = 42

    # --- A. 加载离线生成的归一化轨迹 ---
    print("Loading preprocessed expert trajectories...")
    with open("expert_traj_norm.pkl", 'rb') as f:
        expert_trajs = pickle.load(f)
    print(f"Loaded {len(expert_trajs)} normalized trajectories.")

    # --- B. 加载统计量用于环境同步 ---
    stats = np.load("obs_stat.npz")
    obs_mean = stats['mean']
    obs_std = stats['std']

    # --- C. 初始化向量化环境 ---
    # 我们通过 env_kwargs 把统计量传进环境
    # 创建 4 个环境的向量化集合
    venv = DummyVecEnv([make_env for _ in range(4)])

    # --- D. 设置生成器 (Learner) ---
    learner = PPO(
        env=venv,
        policy=MlpPolicy,
        batch_size=128,
        ent_coef=0.01,
        learning_rate=3e-4,
        n_steps=1024,
        gamma=0.99,
        verbose=1,
        seed=SEED,
    )

    # --- E. 设置奖励网络 ---
    # 因为我们已经手动做了 Z-Score，这里可以不用 RunningNorm，
    # 或者保留它作为二层保险（会自动趋近于 1）
    reward_net = BasicShapedRewardNet(
        observation_space=venv.observation_space,
        action_space=venv.action_space,
        normalize_input_layer=RunningNorm,
    )

    # --- F. 初始化 AIRL 训练器 ---
    airl_trainer = AIRL(
        demonstrations=expert_trajs,
        demo_batch_size=1024,
        gen_replay_buffer_capacity=2048,
        n_disc_updates_per_round=8,
        venv=venv,
        gen_algo=learner,
        reward_net=reward_net,
        allow_variable_horizon=True,
    )

    # --- G. 训练与评估 ---
    print("Evaluating before training...")
    mean_reward_before, _ = evaluate_policy(learner, venv, 10)

    print("Starting AIRL training...")
    airl_trainer.train(total_timesteps=100000)

    print("Evaluating after training...")
    mean_reward_after, _ = evaluate_policy(learner, venv, 10)
    print(f"Improvement: {mean_reward_before} -> {mean_reward_after}")

    learner.save("airl_combat_model_final")