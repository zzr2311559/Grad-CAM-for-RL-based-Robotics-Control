# Grad-CAM for RL-base Robotics Control
A PyTorch implementation of Proximal Policy Optimization (PPO) for visual continuous control tasks (DeepMind Control Suite). This project integrates Grad-CAM to visualize and interpret the agent's focus during decision-making.

> For now we only implemented PPO.

## Installation
Refers to `environments.yml`

## Run the experiment

### Training
To train the agent with visual observations and data augmentation:
```
python algos/ppo/ppo.py \
    --domain_name cartpole \
    --task_name balance \
    --exp_name ppo_1 \
    --seed 42
```
### Visualization
To visualize the attention map of a trained agent:
```
python visualize_gradcam.py
```
## Compatibility
This repository currently supports macOS with MPS acceleration. GPU support is not yet available.