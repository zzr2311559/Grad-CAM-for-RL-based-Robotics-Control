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
from algos.ppo.ppo import ActionRepeatWrapper, FrameStackWrapper

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CURRENT_DIR, 'algos', 'ppo', 'data', 'randomcrop', 'pyt_save', 'model.pt')
DOMAIN_NAME = 'cartpole'
TASK_NAME = 'balance'
DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'


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
        # to 84x84
        cam = cv2.resize(cam, (84, 84)) 
        cam = cam - np.min(cam)
        cam = cam / (np.max(cam) + 1e-8)
        return cam

def main():
    ac = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    ac.eval()
    
    try:
        target_layer = ac.pi.encoder.cnn[4] 
    except:
        target_layer = ac.pi.encoder.convs[3] 

    grad_cam = GradCAM(ac, target_layer)
    
    env = suite.load(domain_name=DOMAIN_NAME, task_name=TASK_NAME)
    
    env = ActionRepeatWrapper(env, repeat=2) 
    env = pixels.Wrapper(env, render_kwargs={'height': 100, 'width': 100, 'camera_id': 0})
    env = FrameStackWrapper(env, num_frames=4)
    
    time_step = env.reset()
    
    print("Visualizing... press 'q' to quit")
    
    while True:
        # preprocessing: (100x100) -> Tensor -> Center Crop -> (84x84)
        obs = time_step.observation['pixels'].transpose(2,0,1).copy()
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        
        obs_tensor_cropped = center_crop(obs_tensor, output_size=84)
        obs_tensor_cropped.requires_grad = True
        
        mask = grad_cam(obs_tensor_cropped)
        
        with torch.no_grad():
            a = ac.act(obs_tensor_cropped)
        
        time_step = env.step(a)
        
        img_full = time_step.observation['pixels'][:,:,-3:] 
        
        h, w, _ = img_full.shape
        sh = (h - 84) // 2
        sw = (w - 84) // 2
        img_cropped = img_full[sh:sh+84, sw:sw+84, :]

        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        
        cam_img = heatmap * 0.4 + np.float32(img_cropped) / 255 * 0.6
        cam_img = cam_img / np.max(cam_img)
        
        cam_img_big = cv2.resize(cam_img, (400, 400), interpolation=cv2.INTER_NEAREST)
        
        cv2.imshow('Grad-CAM', np.uint8(255 * cam_img_big))
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()