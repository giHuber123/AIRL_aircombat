import pickle
import numpy as np
import random
import os
from imitation.data.types import Trajectory


def preprocess_and_save(input_path, output_path, stat_path):
    # 1. 加载原始数据
    with open(input_path, 'rb') as f:
        data = pickle.load(f)

    # 2. 切分 Episodes
    episodes = []
    current_episode = []
    for step in data:
        current_episode.append(step)
        if step['done']:
            episodes.append(current_episode)
            current_episode = []

    print(f"解析完成：共 {len(episodes)} 局数据。")

    # 3. 计算统计量 (建议使用全部数据计算，比随机一局更稳健)
    all_obs = []
    for ep in episodes:
        for s in ep:
            all_obs.append(s['obs'])
        all_obs.append(ep[-1]['next_obs'])

    all_obs = np.array(all_obs)
    mean = np.mean(all_obs, axis=0)
    std = np.std(all_obs, axis=0) + 1e-8

    # 4. 转换并归一化
    norm_trajectories = []
    for ep in episodes:
        obs_list = [(s['obs'] - mean) / std for s in ep]
        obs_list.append((ep[-1]['next_obs'] - mean) / std)  # 补 N+1

        acts_list = [s['action'] for s in ep]
        infos_list = [{} for _ in ep]

        norm_trajectories.append(Trajectory(
            obs=np.array(obs_list, dtype=np.float32),
            acts=np.array(acts_list, dtype=np.float32),
            infos=np.array(infos_list),
            terminal=True
        ))

    # 5. 保存处理后的轨迹对象
    with open(output_path, 'wb') as f:
        pickle.dump(norm_trajectories, f)

    # 6. 保存统计量（非常重要：环境需要用它！）
    np.savez(stat_path, mean=mean, std=std)

    print(f"处理完成！")
    print(f"轨迹已保存至: {output_path}")
    print(f"统计量已保存至: {stat_path}")
    print(f"Obs 均值范围: {np.mean(all_obs, axis=0)[:3]}...")


if __name__ == "__main__":
    preprocess_and_save(
        input_path="expert_traj_test.pkl",
        output_path="expert_traj_norm.pkl",
        stat_path="obs_stat.npz"
    )