import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from dm_control import suite
from dm_control.suite.wrappers import pixels
from collections import deque
from dm_env import specs
import algos.ppo.core as core
import os


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CURRENT_DIR, 'algos', 'ppo', 'data', 'randomcrop', 'pyt_save', 'model.pt') # 替换为你实际的模型路径
DOMAIN_NAME = 'cartpole'
TASK_NAME = 'balance'
DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'


# ✅ 1. 必须复刻 FrameStackWrapper
class FrameStackWrapper:
    def __init__(self, env, num_frames=3):
        self._env = env
        self._num_frames = num_frames
        self._frames = deque([], maxlen=num_frames)
        
        orig_spec = self._env.observation_spec()['pixels']
        new_shape = orig_spec.shape[:-1] + (orig_spec.shape[-1] * num_frames,)
        
        self._obs_spec = specs.BoundedArray(
            shape=new_shape,
            dtype=orig_spec.dtype,
            minimum=0,
            maximum=255,
            name='pixels'
        )

    def _transform_observation(self, time_step):
        obs = time_step.observation
        pixels = obs['pixels']
        
        if len(self._frames) == 0:
            for _ in range(self._num_frames):
                self._frames.append(pixels)
        else:
            self._frames.append(pixels)
            
        stacked_pixels = np.concatenate(list(self._frames), axis=-1)
        new_obs = obs.copy()
        new_obs['pixels'] = stacked_pixels
        return time_step._replace(observation=new_obs)

    def reset(self):
        self._frames.clear()
        time_step = self._env.reset()
        return self._transform_observation(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        return self._transform_observation(time_step)

    def observation_spec(self):
        spec = self._env.observation_spec()
        spec['pixels'] = self._obs_spec
        return spec

    def action_spec(self):
        return self._env.action_spec()
    
    def __getattr__(self, name):
        return getattr(self._env, name)

# ✅ 2. 必须复刻 Center Crop
def center_crop(imgs, output_size=84):
    # imgs: (1, C, H, W)
    h, w = imgs.shape[2], imgs.shape[3]
    start_h = (h - output_size) // 2
    start_w = (w - output_size) // 2
    return imgs[:, :, start_h:start_h + output_size, start_w:start_w + output_size]

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, obs_tensor):
        self.model.zero_grad()
        pi_distribution = self.model.pi._distribution(obs_tensor)
        mu = pi_distribution.loc
        score = mu[0, 0]
        score.backward()
        
        gradients = self.gradients.data.cpu().numpy()[0]
        activations = self.activations.data.cpu().numpy()[0]
        weights = np.mean(gradients, axis=(1, 2))
        
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
            
        cam = np.maximum(cam, 0)
        # 缩放到网络输入的大小 (84x84)
        cam = cv2.resize(cam, (84, 84)) 
        cam = cam - np.min(cam)
        cam = cam / (np.max(cam) + 1e-8)
        return cam

def main():
    # 加载模型
    ac = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    ac.eval()
    
    # Hook 最后一层卷积
    # 注意: 你的网络可能有 4 层卷积 (如果是 RAD Encoder)，或者是 3 层 (Nature CNN)
    # 请根据实际情况调整索引。如果是 Nature CNN，索引通常是 4；如果是 RAD，可能是 6 或 8
    # 这里假设是 Nature CNN (0, 2, 4 是卷积层)
    try:
        target_layer = ac.pi.encoder.cnn[4] 
    except:
         # 如果是 RAD Encoder，可能是这样，你需要 print(ac) 看看结构
        target_layer = ac.pi.encoder.convs[3] # 假设是第4层卷积

    grad_cam = GradCAM(ac, target_layer)
    
    # ✅ 3. 环境初始化：完全复刻训练时的配置
    env = suite.load(domain_name=DOMAIN_NAME, task_name=TASK_NAME)
    
    # 如果训练用了 Action Repeat，这里也要加 (虽然对单步可视化影响不大，但为了物理一致性)
    # env = ActionRepeatWrapper(env, repeat=2) 
    
    # 如果训练用了 100x100 做 Random Crop，这里必须也是 100x100
    env = pixels.Wrapper(env, render_kwargs={'height': 100, 'width': 100, 'camera_id': 0})
    
    # 如果训练用了 FrameStack=4 (根据你的报错推断的)，这里必须是 4
    env = FrameStackWrapper(env, num_frames=4)
    
    time_step = env.reset()
    
    print("开始可视化... 按 'q' 退出")
    
    while True:
        # 预处理: (100x100) -> Tensor -> Center Crop -> (84x84)
        obs = time_step.observation['pixels'].transpose(2,0,1).copy()
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        
        # ✅ 关键：中心裁剪到 84x84，这是模型见过的样子
        obs_tensor_cropped = center_crop(obs_tensor, output_size=84)
        obs_tensor_cropped.requires_grad = True
        
        # 获取热力图
        mask = grad_cam(obs_tensor_cropped)
        
        # 动作推理
        with torch.no_grad():
            a = ac.act(obs_tensor_cropped)
        
        time_step = env.step(a)
        
        # --- 可视化合成 ---
        # 我们只取最新的一帧画面来显示 (obs 有 12 个通道，取后 3 个)
        # 注意：这里的 img 是 100x100 的原图
        img_full = time_step.observation['pixels'][:,:,-3:] 
        
        # 把原图也中心裁剪到 84x84，以便和热力图对齐显示
        h, w, _ = img_full.shape
        sh = (h - 84) // 2
        sw = (w - 84) // 2
        img_cropped = img_full[sh:sh+84, sw:sw+84, :]

        # 热力图伪彩色
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        
        # 叠加
        cam_img = heatmap * 0.4 + np.float32(img_cropped) / 255 * 0.6
        cam_img = cam_img / np.max(cam_img)
        
        # 放大一点显示，不然 84x84 太小了
        cam_img_big = cv2.resize(cam_img, (400, 400), interpolation=cv2.INTER_NEAREST)
        
        cv2.imshow('Grad-CAM', np.uint8(255 * cam_img_big))
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()