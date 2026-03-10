import pickle
import random
import numpy as np


def check_expert_data(pkl_path):
    with open(pkl_path, 'rb') as f:
        trajectory = pickle.load(f)

    # --- 1. 将平铺的列表重新拆分为“局” ---
    episodes = []
    current_ep = []

    for step in trajectory:
        current_ep.append(step)
        if step.get('done', False):
            episodes.append(current_ep)
            current_ep = []

    # 如果最后一段没存 done (异常情况)，也把它加进去
    if current_ep:
        episodes.append(current_ep)

    total_episodes = len(episodes)
    total_steps = len(trajectory)

    print(f"--- 专家数据概览 ---")
    print(f"总计步数: {total_steps}")
    print(f"总计局数: {total_episodes}")
    print(f"平均每局步数: {total_steps / total_episodes:.2f}")
    print("-" * 30)

    # --- 2. 随机抽取一局进行详细检查 ---
    if total_episodes > 0:
        idx = random.randint(0, total_episodes - 1)
        sample_ep = episodes[idx]

        print(f"随机抽取第 {idx} 局 (长度: {len(sample_ep)} 步):")

        # 打印这一局的前 3 步和最后 1 步
        indices_to_show = [0, 1, 2, -1]
        for i in indices_to_show:
            if abs(i) < len(sample_ep):
                step = sample_ep[i]
                prefix = "START" if i == 0 else ("END" if i == -1 else f"STEP {i}")

                # 格式化打印 obs 和 action (只显示前 4 位防止刷屏)

                obs_preview = step['obs'][:4]
                act_preview = step['action']
                done_val = step['done']

                print(f"[{prefix}] Obs: {obs_preview}... | Act: {act_preview} | Done: {done_val}")

        # --- 3. 基础统计检查 ---
        all_obs = np.array([s['obs'] for s in sample_ep])
        print("-" * 30)
        print(f"本局 Obs 均值: {np.mean(all_obs):.4f}")
        print(f"本局 Obs 最大值: {np.max(all_obs):.4f}")
        print(f"本局 Obs 最小值: {np.min(all_obs):.4f}")

    else:
        print("错误：没有找到完整的局（可能缺少 done 信号）")

# 使用方法
# check_expert_data("my_f16_expert_trajectories.pkl")
if __name__ == "__main__":
    check_expert_data("expert_traj_test.pkl")