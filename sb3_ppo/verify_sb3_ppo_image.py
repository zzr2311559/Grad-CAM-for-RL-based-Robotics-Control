import gymnasium as gym
import numpy as np
import os
import shimmy          
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import ResizeObservation

# 解决gym各版本库饮用位置问题
try:
    from gymnasium.wrappers import FrameStackObservation
except ImportError:
    from gymnasium.wrappers import FrameStack as FrameStackObservation

try:
    from gymnasium.wrappers import RenderObservation
except ImportError:
    from gymnasium.wrappers import AddRenderObservation as RenderObservation
        
        

# --- 1. 自定义 Wrapper：只保留像素输入 ---
# SB3 的 CnnPolicy 默认期望输入是 Tensor (C, H, W)，而不是 Dict
class ExtractPixelsWrapper(gym.ObservationWrapper):
    def __init__(self, env, pixel_key='pixels'):
        super().__init__(env)
        self.pixel_key = pixel_key
        # 确保观测空间是 Box 类型 (C, H, W) 或 (H, W, C)
        # RenderObservation 通常把图像放在一个 key 下，我们需要把它提出来
        if isinstance(env.observation_space, gym.spaces.Dict):
            self.observation_space = env.observation_space[pixel_key]
        else:
            # 如果已经是 Box 了 (被其他 wrapper 处理过)，就直接用
            self.observation_space = env.observation_space

    def observation(self, observation):
        if isinstance(observation, dict):
            return observation[self.pixel_key]
        return observation

# --- 2. 环境制造工厂 ---
def make_env(domain_name, task_name, render_mode='rgb_array', height=84, width=84):
    # 使用 shimmy 加载 dm_control 环境
    env_id = f"dm_control/{domain_name}-{task_name}-v0"
    
    env = gym.make(env_id, render_mode=render_mode)
    
    # 步骤 A: 添加渲染图像到观测字典 (Key 通常是 'pixels' 或 'render')
    # 注意: shimmy/gymnasium 的 RenderObservation 默认 key 可能是 'pixels'
    try:
        env = RenderObservation(env) 
    except TypeError:
        # 部分旧版本 wrapper 需要指定 key
        env = RenderObservation(env, key='pixels')
        
    # 步骤 B: 提取 'pixels'，丢弃原有的 state，只保留图像
    # 这样输出就是纯图像 (H, W, C)
    env = ExtractPixelsWrapper(env, pixel_key='pixels')
    
    # 步骤 C: 调整大小到 84x84
    env = ResizeObservation(env, (height, width))
    
    # 步骤 D: 监控器 (记录 log)
    env = Monitor(env)
    
    # 注意：这里不加 FrameStack，我们在外面用 SB3 的 VecFrameStack 加
    # 因为 SB3 的 VecFrameStack 处理 Channel First/Last 更智能
    
    return env

# --- 3. 主训练逻辑 ---
if __name__ == '__main__':
    # 配置
    DOMAIN = 'cartpole'
    TASK = 'balance' # 或者 'swingup'
    TOTAL_TIMESTEPS = 200_000 # 跑 20 万步验证
    N_ENVS = 1 # 你也可以设为 4 或 8 来并行加速 (SB3 对并行支持很好)
    
    # 创建向量化环境
    # DummyVecEnv 适用于单进程调试，SubprocVecEnv 适用于多核并行
    env = DummyVecEnv([lambda: make_env(DOMAIN, TASK) for _ in range(N_ENVS)])
    
    # 关键技巧：Frame Stack (自动堆叠 3 帧，解决速度感知问题)
    # SB3 的 VecFrameStack 会自动处理通道堆叠 (C*k, H, W)
    env = VecFrameStack(env, n_stack=3)
    
    # SB3 会自动处理 VecTransposeImage (把 HWC 转为 CHW)，通常不需要手动加
    
    print(f"Observation Space: {env.observation_space.shape}") # 应该是 (9, 84, 84) -> 3RGB * 3Stack
    
    # 初始化 PPO 模型
    model = PPO(
        "CnnPolicy",  # SB3 的 CnnPolicy 默认就是 NatureCNN
        env,
        verbose=1,
        learning_rate=1e-4,     # 我们之前调优过的学习率
        n_steps=4096,           # buffer size
        batch_size=128,         # 稍微小一点的 batch size 有助于随机性
        n_epochs=10,            # 每次更新过几遍数据
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,          # 加上熵奖励 (我们之前的经验)
        vf_coef=0.5,
        max_grad_norm=0.5,      # 默认开启梯度裁剪 (Actor 和 Critic 都会裁)
        tensorboard_log="./sb3_log/"
    )
    
    print("Start training with SB3...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    
    # 保存
    model.save("sb3_ppo_cartpole_pixel")
    print("Training finished and model saved.")
    
    # --- 4. 验证/观看效果 ---
    # 只有在 GUI 环境下才能看到弹窗，服务器上请跳过
    # obs = env.reset()
    # for i in range(1000):
    #     action, _states = model.predict(obs, deterministic=True)
    #     obs, reward, done, info = env.step(action)
    #     env.render("human")