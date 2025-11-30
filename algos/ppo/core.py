import numpy as np
import scipy.signal
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input: 
        vector x, 
        [x0, 
         x1, 
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]



class CNNbase(nn.Module):
    def __init__(self, obs_shape, hidden_size=512):
        super().__init__()
        
        n_input_channels = obs_shape[0]
        
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # compute the dimension of output automatically
        with torch.no_grad():
            sample_input = torch.zeros(1, *obs_shape)
            output_shape = self.cnn(sample_input).shape
            self.cnn_out_dim = output_shape[1]

        self.linear = nn.Sequential(
            nn.Linear(self.cnn_out_dim, hidden_size),
            nn.ReLU()
        )
        
    def forward(self, x):
        x = x.float() / 255.0
        x = self.cnn(x)
        x = self.linear(x)
        return x
        
class CNNGaussianActor(nn.Module):
    def __init__(self, encoder, hidden_size, act_dim):
        super().__init__()
        self.encoder = encoder
        self.mu_net = nn.Linear(hidden_size, act_dim)
        
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        
    def _distribution(self, obs):
        features = self.encoder(obs)
        mu = self.mu_net(features)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)

    def forward(self, obs, act=None):
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a
    
class CNNCritic(nn.Module):
    def __init__(self, encoder, hidden_size):
        super().__init__()
        self.encoder = encoder
        self.v_net = nn.Linear(hidden_size, 1)

    def forward(self, obs):
        features = self.encoder(obs)
        return torch.squeeze(self.v_net(features), -1)
        
class CNNActorCritic(nn.Module):
    
    def __init__(self, obs_dim, act_dim, hidden_sizes=(512,), activation=nn.Tanh):
        super().__init__()
                    
        self.encoder_pi = CNNbase(obs_dim, hidden_size=hidden_sizes[0])        
        self.encoder_v = CNNbase(obs_dim, hidden_size=hidden_sizes[0])        


        self.pi = CNNGaussianActor(self.encoder_pi, hidden_sizes[0], act_dim[0])
        self.v = CNNCritic(self.encoder_v, hidden_sizes[0])
        
    def step(self, obs):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
        return a.cpu().numpy(), v.cpu().numpy(), logp_a.cpu().numpy()
    
    def act(self, obs):
        return self.step(obs)[0]