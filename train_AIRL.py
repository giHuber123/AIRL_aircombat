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
        ent_coef=0.0,
        learning_rate=0.0005,
        gamma=0.95,
        clip_range=0.1,
        vf_coef=0.1,
        n_epochs=10,
        tensorboard_log="./airl_log/",
        seed=SEED,
    )
    reward_net = BasicShapedRewardNet(
        observation_space=env.observation_space,
        action_space=env.action_space,
        normalize_input_layer=RunningNorm,
    )
    airl_trainer = AIRL(
        demonstrations=expert_trajs,
        demo_batch_size=1024,
        gen_replay_buffer_capacity=4096,
        n_disc_updates_per_round=5,
        venv=venv,
        gen_algo=learner,
        reward_net=reward_net,
        allow_variable_horizon=True,
    )

    airl_trainer.train(1000000)  # Train for 2_000_000 steps to match expert.
    learner.save("airl_combat_model_final")
    # SEED = 42
    #
    # # --- A. 加载离线生成的归一化轨迹 ---
    # print("Loading preprocessed expert trajectories...")
    # with open("expert_traj_norm.pkl", 'rb') as f:
    #     expert_trajs = pickle.load(f)
    # print(f"Loaded {len(expert_trajs)} normalized trajectories.")
    #
    # # --- B. 加载统计量用于环境同步 ---
    # stats = np.load("obs_stat.npz")
    # obs_mean = stats['mean']
    # obs_std = stats['std']
    #
    # # --- C. 初始化向量化环境 ---
    # # 我们通过 env_kwargs 把统计量传进环境
    # # 创建 4 个环境的向量化集合
    # venv = DummyVecEnv([make_env for _ in range(4)])
    #
    # # --- D. 设置生成器 (Learner) ---
    # learner = PPO(
    #     env=venv,
    #     policy=MlpPolicy,
    #     batch_size=128,
    #     ent_coef=0.01,
    #     learning_rate=3e-4,
    #     n_steps=1024,
    #     gamma=0.99,
    #     verbose=1,
    #     seed=SEED,
    # )
    #
    # # --- E. 设置奖励网络 ---
    # # 因为我们已经手动做了 Z-Score，这里可以不用 RunningNorm，
    # # 或者保留它作为二层保险（会自动趋近于 1）
    # reward_net = BasicShapedRewardNet(
    #     observation_space=venv.observation_space,
    #     action_space=venv.action_space,
    #     normalize_input_layer=RunningNorm,
    # )
    #
    # # --- F. 初始化 AIRL 训练器 ---
    # airl_trainer = AIRL(
    #     demonstrations=expert_trajs,
    #     demo_batch_size=1024,
    #     gen_replay_buffer_capacity=2048,
    #     n_disc_updates_per_round=8,
    #     venv=venv,
    #     gen_algo=learner,
    #     reward_net=reward_net,
    #     allow_variable_horizon=True,
    # )
    #
    # # --- G. 训练与评估 ---
    # print("Evaluating before training...")
    # mean_reward_before, _ = evaluate_policy(learner, venv, 10)
    #
    # print("Starting AIRL training...")
    # airl_trainer.train(total_timesteps=100000)
    #
    # print("Evaluating after training...")
    # mean_reward_after, _ = evaluate_policy(learner, venv, 10)
    # print(f"Improvement: {mean_reward_before} -> {mean_reward_after}")
    #
    # learner.save("airl_combat_model_final")