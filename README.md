# Temporal-Difference-Models-with-Value-Function-Based-sampling
Keras implementation of [Temporal Difference Models](https://arxiv.org/abs/1802.09081) by V. Pong et al.(2018) + Value based sampling 

Requirements:
- Keras
- Tensorflow or Theano as backend
- [Gym](https://github.com/openai/gym) - Robotics environment along with [Mujoco](https://github.com/openai/mujoco-py#obtaining-the-binaries-and-license-key) environment

The structure of this code is built on [Keras-rl](https://github.com/keras-rl/keras-rl)

A few tweaks that we did - 
1. Relabelling goals based on expected reward henceforth, with some probability. We found that it lead to faster convergence in the FetchReach environment. 
2. Decayed the 'goal reached' condition radius gradually. It lead to faster convergence as well

