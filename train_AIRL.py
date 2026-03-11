import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
from Single_flight_env import FlightEnv
from net_disc import FighterDiscriminator
from net_policy import FighterPolicyNet
from replay_buffer import AIRLBuffer
import pickle
import os

# --- 1. 网络定义 (基于你之前的需求) ---
# FighterPolicyNet 和 FighterDiscriminator 已经定义好，这里直接实例化

# --- 2. 训练调度器 ---
class AIRLTrainer:
    def __init__(self, env, policy_net, disc_net, expert_buffer):
        self.env = env
        self.policy = policy_net
        self.disc = disc_net
        self.expert_buffer = expert_buffer  # 预先填好的专家经验池
        self.agent_buffer = AIRLBuffer(capacity=10000)  # 刚才讨论的特殊经验池

        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=3e-4)
        self.disc_optimizer = optim.Adam(self.disc.parameters(), lr=1e-4)

        self.gamma = 0.99
        self.seq_len = 5

    def collect_trajectories(self, num_steps):
        """智能体在 JSBSim 环境中采样"""
        state = self.env.reset()
        # 使用 deque 维护 14 维状态序列
        state_deque = deque([state] * self.seq_len, maxlen=self.seq_len)

        for _ in range(num_steps):
            # 将 deque 转换为 [1, 5, 14] 的 Tensor
            state_seq = torch.FloatTensor(np.array(state_deque)).unsqueeze(0)

            # 策略网络输出动作
            with torch.no_grad():
                action = self.policy(state_seq).squeeze(0).numpy()

            # 执行环境步
            next_state, _, truncated, terminated, _ = self.env.step(action)

            # 维护下一个状态序列
            next_state_deque = state_deque.copy()
            next_state_deque.append(next_state)

            # 存入经验池
            self.agent_buffer.add(np.array(state_deque), action, np.array(next_state_deque))

            state_deque = next_state_deque
            if truncated or terminated:
                state = self.env.reset()
                state_deque = deque([state] * self.seq_len, maxlen=self.seq_len)

    def update_discriminator(self, batch_size):
        """更新判别器 (AIRL 的核心：学习奖励函数)"""
        # 1. 采样专家数据和智能体数据
        exp_obs_seq, exp_act, _ = self.expert_buffer.sample(batch_size)
        agt_obs_seq, agt_act, _ = self.agent_buffer.sample(batch_size)

        # 2. 计算判别器输出
        # D(s,a) = exp(f(s,a)) / (exp(f(s,a)) + pi(a|s))
        # 简化版：直接对 f(s,a) 做二分类交叉熵
        exp_logits = self.disc(exp_obs_seq, exp_act)
        agt_logits = self.disc(agt_obs_seq, agt_act)

        # 专家打高分(1)，智能体打低分(0)
        loss_fn = nn.BCEWithLogitsLoss()
        loss = loss_fn(exp_logits, torch.ones_like(exp_logits)) + \
               loss_fn(agt_logits, torch.zeros_like(agt_logits))

        self.disc_optimizer.zero_grad()
        loss.backward()
        self.disc_optimizer.step()
        return loss.item()

    def update_policy(self, batch_size):
        """更新策略网络 (利用判别器生成的奖励进行强化学习)"""
        agt_obs_seq, agt_act, _ = self.agent_buffer.sample(batch_size)

        # 1. 使用判别器生成“伪奖励”
        with torch.no_grad():
            # AIRL 奖励公式: r = log(D) - log(1-D)
            # 在实现中通常直接取判别器的输出值作为奖励
            reward = self.disc(agt_obs_seq, agt_act)

            # 2. 强化学习更新 (这里以简化的策略梯度为例，实际建议用 PPO)
        # 目标：最大化期望奖励
        current_action = self.policy(agt_obs_seq)
        # 这里的损失函数取决于你选用的具体 RL 算法 (如 PPO 的剪切损失)
        policy_loss = -(current_action * reward).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        return policy_loss.item()

    def train(self, total_epochs):
        for epoch in range(total_epochs):
            # 第一步：采样
            self.collect_trajectories(num_steps=200)

            # 第二步：更新判别器
            d_loss = self.update_discriminator(batch_size=64)

            # 第三步：更新策略
            p_loss = self.update_policy(batch_size=64)

            if epoch % 10 == 0:
                print(f"Epoch {epoch}: D_Loss: {d_loss:.4f}, P_Loss: {p_loss:.4f}")
                self.env.render()  # 每一百代存一个 Tacview 文件查看效果




def load_expert_data_to_buffer(buffer, pkl_path, seq_len=5):
    with open(pkl_path, 'rb') as f:
        # 假设 trajectory 是一个巨大的 list，包含了很多局的数据点
        trajectory = pickle.load(f)

    # 记录当前连续帧的计数，只有连续帧达到 seq_len 才能组成一个有效序列
    current_episode_steps = []

    for i in range(len(trajectory)):
        step_data = trajectory[i]
        current_episode_steps.append(step_data)

        # 检查是否满足提取条件
        if len(current_episode_steps) >= seq_len + 1:
            # 1. 提取当前 5 步的 obs 序列 (s)
            obs_seq = np.array([s['obs'] for s in current_episode_steps[-seq_len - 1: -1]])

            # 2. 提取当前动作 (a) - 对应序列最后一步做出的决策
            action = current_episode_steps[-2]['action']

            # 3. 提取下一步的 obs 序列 (s')
            next_obs_seq = np.array([s['obs'] for s in current_episode_steps[-seq_len:]])

            buffer.add(obs_seq, action, next_obs_seq)

        # --- 核心逻辑：检测“局”的终点 ---
        # 如果当前步是 done，或者下一时刻的数据是不连续的，则清空计数器
        if step_data.get('done', False):
            current_episode_steps = []  # 遇到 done，立刻重置，防止跨局切片

    print(f"专家数据加载完毕，当前 Buffer 长度: {len(buffer)}")

# --- 3. 启动训练 ---
if __name__ == "__main__":
    env = FlightEnv()
    expert_buffer = AIRLBuffer(capacity=10000)
    discriminator = FighterDiscriminator()
    policy = FighterPolicyNet()
    trainer = AIRLTrainer(env, policy, discriminator, expert_buffer)
    trainer.train(1000)