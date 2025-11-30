import sys
import os
import os.path as osp

current_dir = osp.dirname(osp.abspath(__file__))
root_dir = osp.join(current_dir, '..', '..')
sys.path.append(root_dir)

from collections import deque
import numpy as np
import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from dm_control import suite
from dm_control.suite.wrappers import pixels
from dm_env import specs
import time
import algos.ppo.core as core
from utils.logx import EpochLogger


import torch
import torch.nn.functional as F

def random_crop(imgs, output_size=84):
    """
    Input: imgs (N, C, H, W)
    Output: cropped (N, C, output_size, output_size)
    """
    n, c, h, w = imgs.shape
    crop_max = h - output_size
    
    w1 = torch.randint(0, crop_max + 1, (n,), device=imgs.device).float()
    h1 = torch.randint(0, crop_max + 1, (n,), device=imgs.device).float()

    y = torch.arange(output_size, device=imgs.device).float()
    x = torch.arange(output_size, device=imgs.device).float()
    grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
    
    grid_x = grid_x.unsqueeze(0) + w1.view(-1, 1, 1)
    grid_y = grid_y.unsqueeze(0) + h1.view(-1, 1, 1)
    
    norm_grid_x = 2.0 * grid_x / (w - 1.0) - 1.0
    norm_grid_y = 2.0 * grid_y / (h - 1.0) - 1.0
    
    grid = torch.stack((norm_grid_x, norm_grid_y), dim=-1)
    
    cropped = F.grid_sample(imgs, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
    
    return cropped

def center_crop(imgs, output_size=84):

    if len(imgs.shape) == 3:
        h, w = imgs.shape[1], imgs.shape[2]
        start_h = (h - output_size) // 2
        start_w = (w - output_size) // 2
        return imgs[:, start_h:start_h + output_size, start_w:start_w + output_size]
    else:
        h, w = imgs.shape[2], imgs.shape[3]
        start_h = (h - output_size) // 2
        start_w = (w - output_size) // 2
        return imgs[:, :, start_h:start_h + output_size, start_w:start_w + output_size]

class ActionRepeatWrapper:
    def __init__(self, env, repeat=2):
        self._env = env
        self._repeat = repeat

    def __getattr__(self, name):
        return getattr(self._env, name)

    def step(self, action):
        total_reward = 0.0
        for _ in range(self._repeat):
            time_step = self._env.step(action)
            reward = time_step.reward
            if reward is not None:
                total_reward += reward
            
            if time_step.last():
                break
        
        return time_step._replace(reward=total_reward)

    def reset(self):
        return self._env.reset()

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

class FrameStackWrapper:
    def __init__(self, env, num_frames=4):
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
def statistics_scalar(x, with_min_and_max=False):
    x = np.array(x, dtype=np.float32)
    mean = np.mean(x)
    std = np.std(x)
    if with_min_and_max:
        return mean, std, np.min(x), np.max(x)
    return mean, std

class PPOBuffer:
    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95, device='cpu'):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size
        self.device = device

    def store(self, obs, act, rew, val, logp):
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = core.discount_cumsum(deltas, self.gamma * self.lam)
        self.ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)[:-1]
        self.path_start_idx = self.ptr

    def get(self):
        assert self.ptr == self.max_size
        self.ptr, self.path_start_idx = 0, 0
        adv_mean, adv_std = statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf)
        
        return {k: torch.as_tensor(v, dtype=torch.float32).to(self.device) for k,v in data.items()}

def ppo(env_fn, actor_critic=core.CNNActorCritic, hidden_sizes=(512,), seed=0, 
        steps_per_epoch=4000, epochs=50, gamma=0.99, clip_ratio=0.2, pi_lr=1e-4,
        vf_lr=1e-3, train_pi_iters=80, train_v_iters=80, lam=0.97, max_ep_len=1000,
        target_kl=0.02, logger_kwargs=dict(), save_freq=1):

    
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon) acceleration!")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    torch.manual_seed(seed)
    np.random.seed(seed)

    # 2. 环境初始化
    env = env_fn()
    env = ActionRepeatWrapper(env, repeat=2)
    env = pixels.Wrapper(env, render_kwargs={'height': 100, 'width': 100, 'camera_id': 0})
    env = FrameStackWrapper(env, num_frames=4)
    
    obs_spec = env.observation_spec()['pixels']
    act_spec = env.action_spec()
    
    
    #obs_dim = (obs_spec.shape[2], obs_spec.shape[0], obs_spec.shape[1])
    c = obs_spec.shape[2] 
    h_env, w_env = obs_spec.shape[0], obs_spec.shape[1]
    buffer_obs_dim = (c, h_env, w_env) # (12, 100, 100)
    network_obs_dim = (c, 84, 84) 
    
    act_dim = act_spec.shape
    # ac = actor_critic(obs_dim, act_dim, hidden_sizes)
    
    ac = actor_critic(network_obs_dim, act_dim, hidden_sizes)

    
    ac.to(device)

    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.v])
    logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n'%var_counts)

    # buf = PPOBuffer(obs_dim, act_dim, steps_per_epoch, gamma, lam, device=device)

    buf = PPOBuffer(buffer_obs_dim, act_dim, steps_per_epoch, gamma, lam, device=device)
    
    # Loss Functions
    def compute_loss_pi(data):
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']
        
        obs = random_crop(obs, output_size=84)
        
        pi, logp = ac.pi(obs, act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio) * adv
        
        # loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()
        # approx_kl = (logp_old - logp).mean().item()
        # ent = pi.entropy().mean().item()
        
        entropy = pi.entropy().mean()
        ent_coef = 0.01
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean() - ent_coef * entropy
        approx_kl = (logp_old - logp).mean().item()
        ent = entropy.item()         
        
        
        clipped = ratio.gt(1+clip_ratio) | ratio.lt(1-clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        return loss_pi, dict(kl=approx_kl, ent=ent, cf=clipfrac)

    def compute_loss_v(data):
        obs, ret = data['obs'], data['ret']
        
        obs = random_crop(obs, output_size=84)
        
        return ((ac.v(obs) - ret)**2).mean()

    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    vf_optimizer = Adam(ac.v.parameters(), lr=vf_lr)

    logger.setup_pytorch_saver(ac)

    def update():
        data = buf.get()

        pi_l_old, pi_info_old = compute_loss_pi(data)
        pi_l_old = pi_l_old.item()
        v_l_old = compute_loss_v(data).item()

        # Train Policy
        for i in range(train_pi_iters):
            pi_optimizer.zero_grad()
            loss_pi, pi_info = compute_loss_pi(data)
            kl = pi_info['kl']
            if kl > 1.5 * target_kl:
                logger.log('Early stopping at step %d due to reaching max kl.'%i)
                break
            loss_pi.backward()
            
            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(ac.pi.parameters(), max_norm=0.5)
            pi_optimizer.step()

        logger.store(StopIter=i)

        # Train Value
        for i in range(train_v_iters):
            vf_optimizer.zero_grad()
            loss_v = compute_loss_v(data)
            loss_v.backward()
            
            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(ac.v.parameters(), max_norm=0.5)
            vf_optimizer.step()

        logger.store(LossPi=pi_l_old, LossV=v_l_old,
                     KL=pi_info['kl'], Entropy=pi_info_old['ent'], ClipFrac=pi_info['cf'],
                     DeltaLossPi=(loss_pi.item() - pi_l_old),
                     DeltaLossV=(loss_v.item() - v_l_old))

    start_time = time.time()
    
    time_step = env.reset()
    o = time_step.observation['pixels'].transpose(2,0,1).copy()
    
    ep_ret, ep_len = 0, 0

    for epoch in range(epochs):
        for t in range(steps_per_epoch):
            
            # obs_tensor = torch.as_tensor(o, dtype=torch.float32).unsqueeze(0).to(device)
            obs_full = torch.as_tensor(o, dtype=torch.float32).unsqueeze(0).to(device)
            obs_network_input = center_crop(obs_full, output_size=84)
            
            a, v, logp = ac.step(obs_network_input)
            # a, v, logp = ac.step(obs_tensor)

            time_step = env.step(a[0])
            next_o = time_step.observation['pixels'].transpose(2,0,1).copy()
            
            r = time_step.reward
            if r is None: r = 0.0
            
            d = time_step.last() 
            
            ep_ret += r
            ep_len += 1
            
            buf.store(o, a[0], r, v[0], logp[0])
            logger.store(VVals=v)
            
            o = next_o

            timeout = ep_len == max_ep_len
            terminal = d or timeout
            epoch_ended = t == steps_per_epoch-1

            if terminal or epoch_ended:
                if epoch_ended and not(terminal):
                    print('Warning: trajectory cut off by epoch at %d steps.'%ep_len, flush=True)
                
                if timeout or epoch_ended:
                    # obs_tensor = torch.as_tensor(o, dtype=torch.float32).unsqueeze(0).to(device)
                    # _, v, _ = ac.step(obs_tensor)
                    
                    obs_tensor = torch.as_tensor(o, dtype=torch.float32).unsqueeze(0).to(device)
                    obs_tensor_cropped = center_crop(obs_tensor, output_size=84)
                    _, v, _ = ac.step(obs_tensor_cropped)
                else:
                    v = 0
                buf.finish_path(v)
                
                if terminal:
                    logger.store(EpRet=ep_ret, EpLen=ep_len)
                
                time_step = env.reset()
                o = time_step.observation['pixels'].transpose(2,0,1).copy()
                ep_ret, ep_len = 0, 0

        if (epoch % save_freq == 0) or (epoch == epochs-1):
            logger.save_state({}, None) 
            pass

        update()

        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('VVals', with_min_and_max=True)
        logger.log_tabular('TotalEnvInteracts', (epoch+1)*steps_per_epoch)
        logger.log_tabular('LossPi', average_only=True)
        logger.log_tabular('LossV', average_only=True)
        logger.log_tabular('DeltaLossPi', average_only=True)
        logger.log_tabular('DeltaLossV', average_only=True)
        logger.log_tabular('Entropy', average_only=True)
        logger.log_tabular('KL', average_only=True)
        logger.log_tabular('ClipFrac', average_only=True)
        logger.log_tabular('StopIter', average_only=True)
        logger.log_tabular('Time', time.time()-start_time)
        logger.dump_tabular()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain_name', type=str, default='cartpole')
    parser.add_argument('--task_name', type=str, default='balance')
    parser.add_argument('--hid', type=tuple, default=(512,))
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--exp_name', type=str, default='ppo_dmc_cnn')
    args = parser.parse_args()


    output_dir = os.path.join('data', args.exp_name)
    logger_kwargs = dict(output_dir=output_dir, exp_name=args.exp_name)

    ppo(lambda : suite.load(domain_name=args.domain_name, task_name=args.task_name), 
        actor_critic=core.CNNActorCritic,
        hidden_sizes=args.hid, gamma=args.gamma, 
        seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs,
        logger_kwargs=logger_kwargs)