import numpy as np
from stable_baselines3 import PPO
from Aircombat_env_for_AIRL import FlightEnv

def _test_model():
    # 1. 加载统计量（必须与训练时使用的一模一样）
    stats = np.load("obs_stat.npz")
    obs_mean = stats['mean']
    obs_std = stats['std']

    # 2. 实例化环境并传入统计量
    # 注意：测试时不需要 Wrapper，直接用类即可
    env = FlightEnv(norm_mean=obs_mean, norm_std=obs_std)

    # 3. 加载训练好的模型
    # 确保路径与你保存时一致
    model = PPO.load("airl_combat_model_final")

    obs, info = env.reset()

    # 记录总奖励（虽然环境返回0，但如果你想看飞行时长，可以累加步数）
    total_steps = 0

    for _ in range(1000):
        # 4. 补全 action 获取逻辑
        # deterministic=True 表示使用最可能的动作，而不是带随机扰动的动作
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        # 5. 可选：调用你的 render 函数生成 Tacview 文件查看轨迹
        env.render()

        total_steps += 1

        if terminated or truncated:
            print(f"Episode finished. Steps: {total_steps}, Reason: {'Terminated' if terminated else 'Truncated'}")
            break

    # 测试结束关闭环境（保存 ACMI 文件）
    env.close()


if __name__ == "__main__":
    _test_model()