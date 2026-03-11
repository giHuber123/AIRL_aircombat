import jsbsim
import numpy as np
from jsbsim import FGFDMExec
import gymnasium as gym
import datetime
import pymap3d
import pickle  # 记得在文件顶部导入
from pyarrow import float32


class FlightEnv(gym.Env):
    def __init__(self, norm_mean=None, norm_std=None):
        self.fdm = FGFDMExec('jsbsim-master')
        self.fdm.set_debug_level(0)
        self.fdm.load_model('f16')
        self.fdm.set_dt(1 / 60)

        self.current_step = 0
        self.acmi_file = None
        self.acm_filename = f"flight_1.acmi"

        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32
        )

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(12,),
            dtype=np.float32
        )

        # 加载对手轨迹数据
        try:
            with open('Manuver_traj.pkl', 'rb') as f:
                self.opp_trajectories = pickle.load(f)
            print(f"成功加载对手轨迹，共 {len(self.opp_trajectories)} 帧")
        except FileNotFoundError:
            self.opp_trajectories = []
            print("警告：未找到 Manuver_traj.pkl")

        self.opp_ptr = 0  # 轨迹指针，从0开始

        self.norm_mean = norm_mean
        self.norm_std = norm_std

        self.reset()

    def reset(self):
        self.fdm["ic/h-sl-ft"] = 20000
        self.fdm['ic/lat-geod-deg'] = 60.1
        self.fdm["ic/psi-true-deg"] = 180.0
        self.fdm['ic/long-gc-deg'] = 120.0
        self.fdm['ic/vt-fps'] = 800
        self.fdm.run_ic()

        self.fdm["fcs/left-aileron-cmd-norm"] = 0.0
        self.fdm["fcs/right-aileron-cmd-norm"] = 0.0
        self.fdm["fcs/elevator-cmd-norm"] = 0.0
        self.fdm["fcs/rudder-cmd-norm"] = 0.0
        self.fdm["fcs/throttle-cmd-norm"] = 0.8  # 维持平飞的推力

        self.opp_ptr = 0  # 重置指针
        self.current_step = 0
        obs = self.get_observation()
        return obs

    def get_observation(self):
        '''
                norm_obs[0] = delta_value[0] / 1000          #  0. ego delta altitude  (unit: 1km)
                norm_obs[1] = in_range_rad(delta_value[1])   #  1. ego delta heading   (unit rad)
                norm_obs[2] = delta_value[2] / 340           #  2. ego delta velocities_u  (unit: mh)
                norm_obs[3] = obs[9] / 5000                  #  3. ego_altitude (unit: km)
                norm_obs[4] = np.sin(obs[3])                 #  4. ego_roll_sin
                norm_obs[5] = np.cos(obs[3])                 #  5. ego_roll_cos
                norm_obs[6] = np.sin(obs[4])                 #  6. ego_pitch_sin
                norm_obs[7] = np.cos(obs[4])                 #  7. ego_pitch_cos
                norm_obs[8] = obs[5] / 340                   #  8. ego_v_x   (unit: mh)
                norm_obs[9] = obs[6] / 340                   #  9. ego_v_y    (unit: mh)
                norm_obs[10] = obs[7] / 340                  #  10. ego_v_z    (unit: mh)
                norm_obs[11] = obs[8] / 340                  #  11. ego_vc        (unit: mh)
                '''

        delta_altitude, delta_heading, delta_velocity = self.get_delta_value()
        obs = np.zeros(12)
        obs[0] = delta_altitude
        obs[1] = self.in_range_rad(delta_heading)
        obs[2] = delta_velocity
        obs[3] = self.fdm['position/h-sl-ft'] * 0.3048
        roll = self.fdm['attitude/phi-rad']
        pitch = self.fdm['attitude/theta-rad']
        obs[4] = np.sin(roll)
        obs[5] = np.cos(roll)
        obs[6] = np.sin(pitch)
        obs[7] = np.cos(pitch)

        obs[8] = self.fdm['velocities/u-fps'] * 0.3048
        obs[9] = self.fdm['velocities/v-fps'] * 0.3048
        obs[10] = self.fdm['velocities/w-fps'] * 0.3048
        obs[11] = self.fdm['velocities/vc-fps'] * 0.3048

        if self.norm_mean is not None:
            normalized_obs = (obs - self.norm_mean) / self.norm_std
            print(f"**************************")
            print(f"step:{self.current_step}")
            print(f"raw_obs:{obs}")
            print(f"normalized_obs:{normalized_obs}")
            return normalized_obs.astype(np.float32)

        return obs

    def step(self, action):
        self.fdm['fcs/aileron-cmd-norm'] = action[0]
        self.fdm['fcs/elevator-cmd-norm'] = action[1]
        self.fdm['fcs/rudder-cmd-norm'] = action[2]
        self.fdm['fcs/throttle-cmd-norm'] = (action[3] + 1.0) * 0.5
        for _ in range(12):
            self.fdm.run()
        self.current_step += 1
        self.opp_ptr += 1
        '''

        读取敌机csv数据
        经纬高
        velocities/u-fps
        '''
        obs = self.get_observation()

        reward = 0
        truncated = False
        terminated = False
        if self.current_step >= 1000:
            # print(f"truncated")
            truncated = True

        if self.fdm['position/h-sl-ft'] * 0.3048 <= 1000:
            # print(f"terminated")
            terminated = True
        return obs, reward, truncated, terminated, {}

    def get_delta_value(self):

        if self.opp_ptr < len(self.opp_trajectories):
            opp_data = self.opp_trajectories[self.opp_ptr]
            # 假设你的 pkl 中 obs 的顺序是之前定义的：[..., alt, sin_roll, ...]
            # 或者根据你保存时的 raw_obs 结构提取
            # 这里演示从 opp_data['obs'] 中提取（假设 index 9 是高度，具体需对应你的保存逻辑）
            opp_raw_obs = opp_data['obs']

            opp_alt = opp_raw_obs[0]  # 示例索引
            opp_lat = opp_raw_obs[1]  # 示例索引
            opp_lon = opp_raw_obs[2]  # 示例索引
            opp_v = opp_raw_obs[3]  # 示例速度 u-mps
        else:
            # 如果轨迹读完了，对手保持最后一帧的状态
            opp_data = self.opp_trajectories[-1]
            opp_raw_obs = opp_data['obs']
            opp_alt, opp_lat, opp_lon, opp_v = opp_raw_obs[0], opp_raw_obs[1], opp_raw_obs[2], opp_raw_obs[3]

        ego_altitude = self.fdm['position/h-sl-ft']
        opp_altitude = opp_alt
        delta_altitude = opp_altitude - ego_altitude * 0.3048

        opp_latitude = opp_lat
        opp_longitude = opp_lon
        opp_position = self.get_position(opp_latitude, opp_longitude, opp_altitude)

        ego_latitude = self.fdm['position/lat-geod-deg']
        ego_longitude = self.fdm['position/long-gc-deg']
        ego_position = self.get_position(ego_latitude, ego_longitude, ego_altitude)

        opp_x, opp_y, _ = opp_position
        ego_x, ego_y, _ = ego_position

        ego_vx, ego_vy, ego_vz = self.get_velocity()
        ego_v = np.linalg.norm([ego_vx, ego_vy])

        delta_x, delta_y = opp_x - ego_x, opp_y - ego_y
        R = np.linalg.norm([delta_x, delta_y])
        proj_dist = delta_x * ego_vx + delta_y * ego_vy
        ego_AO = np.arccos(np.clip(proj_dist / (R * ego_v + 1e-8), -1, 1))
        cross_product = np.cross([ego_vx, ego_vy, 0], [delta_x, delta_y, 0])
        side_flag = np.sign(cross_product[2])  # 取 Z 轴分量的正负号

        delta_heading = ego_AO * side_flag

        ego_v = self.fdm['velocities/u-fps'] * 0.3048
        delta_velocity = opp_v - ego_v

        return np.array([delta_altitude, delta_heading, delta_velocity])

        '''delta_heading
        1. 确定平面相对位置向量
        2. 计算投影距离与夹角
        3. 确定左右转向
        4. 最终输出：有符号的航向偏差
        '''

    def get_velocity(self):
        v_north = self.fdm['velocities/v-north-fps'] * 0.3048
        v_east = self.fdm['velocities/v-east-fps'] * 0.3048
        v_down = self.fdm['velocities/v-down-fps'] * 0.3048
        return np.array([v_north, v_east, v_down])

    def get_position(self, lat, lon, alt, lat0=60.0, lon0=120.0, alt0=0):
        n, e, d = pymap3d.geodetic2ned(lat, lon, alt, lat0, lon0, alt0)
        return np.array([n, e, -d])

    def in_range_rad(self, angle):
        """ Given an angle in rads, normalises in (-pi, pi] """
        angle = angle % (2 * np.pi)
        if angle > np.pi:
            angle -= 2 * np.pi
        return angle

    def render(self):
        # 1. 首次调用时初始化文件并写入 Header
        if self.acmi_file is None:
            self.acmi_file = open(self.acm_filename, "w", encoding="utf-8-sig")
            self.acmi_file.write("FileType=text/acmi/tacview\n")
            self.acmi_file.write("FileVersion=2.1\n")
            self.acmi_file.write("0,ReferenceTime=2024-05-01T00:00:00Z\n")
            # 定义飞机对象 (ID=1)
            self.acmi_file.write("1,Name=F-16,Type=Air+FixedWing,Color=Blue\n")

        # 2. 获取当前仿真时间
        sim_time = self.fdm.get_sim_time()

        # 3. 获取位置信息 (注意单位转换)
        lat = self.fdm['position/lat-geod-deg']
        lon = self.fdm['position/long-gc-deg']
        alt = self.fdm['position/h-sl-ft'] * 0.3048  # Tacview 默认高度单位是米

        # 4. 获取姿态信息 (Tacview 使用度数)
        # 注意：Tacview 的坐标映射可能需要根据版本微调，标准映射如下：
        pitch = self.fdm['attitude/theta-deg']
        roll = self.fdm['attitude/phi-deg']
        hdg = self.fdm['attitude/psi-deg']

        v_u = self.fdm['velocities/u-fps']
        v_v = self.fdm['velocities/v-fps']
        # 5. 写入 ACMI 数据行
        # 格式: #时间戳 \n ID,T=经度|纬度|高度,Coord=俯仰|滚转|偏航
        self.acmi_file.write(f"#{sim_time:.2f}\n")
        self.acmi_file.write(f"1,T={lon}|{lat}|{alt}|{roll}|{pitch}|{hdg}\n")

        # 确保数据实时写入磁盘
        self.acmi_file.flush()

    def close(self):
        # 1. 保存并关闭 ACMI 文件
        if self.acmi_file:
            self.acmi_file.close()
            self.acmi_file = None
            print(f"ACMI record saved to {self.acm_filename}")

        # 2. 显式清理 JSBSim 实例
        if hasattr(self, 'fdm') and self.fdm:
            # 在某些 JSBSim 版本中，显式删除引用有助于释放底层的 C++ 内存
            del self.fdm
            self.fdm = None

        print("Flight environment resources released.")


if __name__ == "__main__":
    stats = np.load("obs_stat.npz")
    obs_mean = stats['mean']
    obs_std = stats['std']
    print(f"obs_mean: {obs_mean}, obs_std: {obs_std}")
    env = FlightEnv(obs_mean, obs_std)
    obs = env.reset()

    while True:
        action = env.action_space.sample()
        obs, reward, truncated, terminated, info = env.step(action)
        if truncated:
            print("truncated")
            break
        if terminated:
            print("terminated")
            break