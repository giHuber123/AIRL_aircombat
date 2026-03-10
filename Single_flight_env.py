import jsbsim
import numpy as np
from jsbsim import FGFDMExec
import gymnasium as gym
import datetime

from pyarrow import float32


class FlightEnv(gym.Env):
    def __init__(self):
        self.fdm = FGFDMExec('jsbsim-master')
        self.fdm.set_debug_level(0)
        self.fdm.load_model('f16')
        self.fdm.set_dt(1/60)

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
            shape=(14,),
            dtype=np.float32
        )

        self.reset()

    def reset(self):
        self.fdm["ic/h-sl-ft"] = 20000
        self.fdm['ic/lat-geod-deg'] = 60.0
        self.fdm["ic/psi-true-deg"] = 180.0
        self.fdm['ic/long-gc-deg'] = 120.0
        self.fdm['ic/vt-fps'] = 800
        self.fdm.run_ic()

        self.fdm["fcs/left-aileron-cmd-norm"] = 0.0
        self.fdm["fcs/right-aileron-cmd-norm"] = 0.0
        self.fdm["fcs/elevator-cmd-norm"] = 0.0
        self.fdm["fcs/rudder-cmd-norm"] = 0.0
        self.fdm["fcs/throttle-cmd-norm"] = 0.8  # 维持平飞的推力

        self.current_step = 0
        obs = self.get_observation()
        return obs

    def get_observation(self):
        '''经纬高'''
        altitude = self.fdm['position/h-sl-ft'] #ft
        latitude = self.fdm['position/lat-geod-rad']
        longitude = self.fdm['position/long-gc-rad']

        '''俯仰角theta 滚转角phi 航向角psi'''
        theta = self.fdm['attitude/theta-rad']
        phi = self.fdm['attitude/phi-rad']
        psi = self.fdm['attitude/psi-rad']
        #需要三角函数处理

        '''攻角（过大过小导致失速） 最大/最小攻角 侧滑角'''
        alpha = self.fdm['aero/alpha-rad']
        alpha_max = self.fdm['aero/alpha-max-rad']
        alpha_min = self.fdm['aero/alpha-min-rad']
        beta = self.fdm['aero/beta-rad']

        '''舵面值 速度值'''
        aileron_right = self.fdm['fcs/right-aileron-pos-norm']
        aileron_left = self.fdm['fcs/left-aileron-pos-norm']
        elevator = self.fdm['fcs/elevator-pos-norm']
        rudder = self.fdm['fcs/rudder-pos-norm']
        v = self.fdm['velocities/vt-fps']

        '''机体系速度分量u前 v右 w下'''
        v_u = self.fdm['velocities/u-fps']
        v_v = self.fdm['velocities/v-fps']
        v_w = self.fdm['velocities/w-fps']

        '''NED系速度分量'''
        v_north = self.fdm['velocities/v-north-fps']
        v_east = self.fdm['velocities/v-east-fps']
        v_down = self.fdm['velocities/v-down-fps']

        '''机体坐标系加速度'''
        u_dot = self.fdm['accelerations/udot-ft_sec2']
        v_dot = self.fdm['accelerations/vdot-ft_sec2']
        w_dot = self.fdm['accelerations/wdot-ft_sec2']

        '''俯仰q/滚转p/偏航r角速度'''
        roll_rate = self.fdm['velocities/p-rad_sec']
        pitch_rate = self.fdm['velocities/q-rad_sec']
        yaw_rate = self.fdm['velocities/r-rad_sec']

        '''角加速度'''
        p_dot = self.fdm['accelerations/pdot-rad_sec2']  # 滚转角加速度
        q_dot = self.fdm['accelerations/qdot-rad_sec2']  # 俯仰角加速度
        r_dot = self.fdm['accelerations/rdot-rad_sec2']  # 偏航角加速度

        '''过载'''
        G_x = self.fdm['accelerations/n-pilot-x-norm'] #x轴归一化过载
        G_y = self.fdm['accelerations/n-pilot-y-norm']
        G_z = self.fdm['accelerations/n-pilot-z-norm']

        obs = np.zeros(14)

        obs = np.array([altitude,
                        np.sin(theta), np.cos(theta), np.sin(phi), np.cos(phi), np.sin(psi), np.cos(psi), #俯仰滚转偏航
                        roll_rate, pitch_rate, yaw_rate,
                        alpha, beta, #攻角侧滑角
                        aileron_right, aileron_left, elevator, rudder, #实际舵面值
                        v, #速度
                        G_x, G_y, G_z #过载
                        ], dtype=np.float32)
        return obs

    def step(self, action):
        self.fdm['fcs/aileron-cmd-norm'] = action[0]
        self.fdm['fcs/elevator-cmd-norm'] = action[1]
        self.fdm['fcs/rudder-cmd-norm'] = action[2]
        self.fdm['fcs/throttle-cmd-norm'] = action[3] #(action[3] + 1.0) * 0.5
        for _ in range(12):
            self.fdm.run()
        self.current_step += 1
        obs = self.get_observation()
        reward = 0
        truncated = False
        terminated = False
        if self.current_step >= 10000:
            print(f"truncated")
            truncated = True

        if self.fdm['position/h-sl-ft'] * 0.3048 <= 1000:
            print(f"terminated")
            terminated = True
        return obs, reward, truncated, terminated, {}

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
