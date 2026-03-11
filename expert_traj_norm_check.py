import pickle
import random
import numpy as np


def check_norm_expert_data(pkl_path):
    with open(pkl_path, 'rb') as f:
        # 现在的 trajectories 是一个 List[Trajectory]
        trajectories = pickle.load(f)

    total_episodes = len(trajectories)
    # 计算总步数：累加每个轨迹的动作数量
    total_steps = sum(len(t.acts) for t in trajectories)

    print(f"--- 归一化专家数据概览 (Trajectory 对象) ---")
    print(f"总计步数: {total_steps}")
    print(f"总计局数: {total_episodes}")
    print(f"平均每局步数: {total_steps / total_episodes:.2f}")
    print("-" * 30)

    if total_episodes > 0:
        # 随机抽取一局
        idx = random.randint(0, total_episodes - 1)
        traj = trajectories[idx]

        print(f"随机抽取第 {idx} 局 (动作长度: {len(traj.acts)} 步, 状态长度: {len(traj.obs)} 步):")

        # 打印前 3 步
        for i in range(min(3, len(traj.acts))):
            obs = traj.obs[i]  # 取前4位
            act = traj.acts[i]
            print(f"[STEP {i}] Obs: {obs} | Act: {act}")

        # 打印最后一步 (终点状态)
        print(f"[FINAL OBS] Obs: {traj.obs[-1][:4]}...")

        # --- 3. 基础统计检查 ---
        # 现在的 obs 已经是归一化后的了
        print("-" * 30)
        print(f"本局 Obs 均值: {np.mean(traj.obs):.4f}")
        print(f"本局 Obs 最大值: {np.max(traj.obs):.4f}")
        print(f"本局 Obs 最小值: {np.min(traj.obs):.4f}")

        # 检查是否真的归一化了（均值应接近0，且数值不会是 6000 这种大数）
        if np.max(np.abs(traj.obs)) > 50:
            print("⚠️ 警告：检测到较大数值，请确认归一化是否生效！")
    else:
        print("错误：pkl 文件中没有轨迹对象")


if __name__ == "__main__":
    check_norm_expert_data("expert_traj_norm.pkl")