---
layout: page
title: CMPUT 503 grad project
permalink: 412-projects/grad/
date: 2025-02-12 10:33:00 -0000
---
Team member: Truong-Giang Pham

Github : [link](https://github.com/TruongGiangPham-1/CMPUT503FinalProject/tree/main)

## Introduction
Reinforcement learning methods have been used to control wheeled robots. The purpose of this grad project is to gain hands-on experience with the capabilities of the Proximal Policy Optimization (PPO) algorithm in learning and performing lane-following tasks (Schulman et al, 2017). The initial goal was to compare it with a PID controller implementation, but that part was omitted from this project.

For this project, I worked with Gym-Duckietown, a Duckietown simulator built on OpenAI’s Gym. This simulator allows full customization and simulation of the entire Duckietown environment. Using the simulator not only alleviates the challenges of working with real robots, but also enables large-scale generation of observations for training reinforcement learning agents.

TL;DR: I originally planned to do a comparative study between a PPO agent and a PID controller agent in Gym-Duckietown. But so far, I only managed to get the PPO agent working. So, there’s no comparative study—just a (bad) showcase of a PPO agent in Gym-Duckietown!

## Method
### 1. Environment
Gym-Duckietown is a fast simulator that follows the Gymnasium convention.
At each time step t, the environment receives an action in the form of [velocity, omega], where velocity controls the forward speed of the Duckiebot, and omega controls the turning angle.

After applying the action, the simulator outputs an observation as a 480×640×3 pixel RGB image.
Each action results in a small incremental movement of the bot within the simulation. While the exact distance moved per time step isn’t quantified, the effect is visually observable.

The simulator truncates the episode after 10,000 time steps or terminates it early if the Duckiebot goes out of bounds.

The reward is a linear combination of these factors.
1. the relative distance between the bot and the center of the right lane
2. the alignment of the bot direction with the curve tangent.
3. Wether or not the bot collided witht the lane outline. 

Some hardcoded rewards
1. -1000 if invalid pose (? unsure how is it computed). 
2. 0 if episode done



### 2. RL algorithm
We train a PPO agent.
The implementation is from `ppo_continous_action.py` from [clearnrl](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_continuous_action.py).

I use a standard Atari-based Convolutional neurla network (CNN) for learning representation based on 
the images.

Here is the model architecture.

| CNN Layers  | Parameters |
| ------------- | ------------- |
| Conv2d  | kernel=8, in_channel=4, out_channel=32 |
| Conv2d  | kernel=3, in_channel_32, out_channel=64  |
| Linear  | kernel=3, in_channel_64, out_channel_64  |

Then I do 2 linear into actor and critic respectively.

Here are the PPO hyperparameters

| params  | value |
| ------------- | ------------- |
| total_timesteps  | 1e6 |
| batch_size  | 2048 |
| mini batch_size  | 2048 / 32 |
| $$\gamma$$  |   0.99 |
| GAE $$\lambda$$  |   0.95 |
| update epoch  | 10 |
| clip_coef  | 0.2 |
| entropy_coef  | 0.1 |
| entropy_coef  | 0.5 |

The exploration is just done using the entropy term.

The reward plots are generated by doing an evaluation phase every K update steps.

I do 1 million simulation steps. I do a PPO update every 2028 simulation steps. 
Therefore, there are  ~ 1 million / 2048 = 488 total PPO updates. 
An evaluation was conducted every 10 PPO updates, leading to a total of 48 evaluations throughout the training process.

See my evaluation implementation [here](https://github.com/TruongGiangPham-1/CMPUT503FinalProject/blob/b2ad23cd2b8f44ad6b1dbe2e28fb0e3118c635f7/utils.py#L81)

### 3. Image Pre-Processing
I mimic image processing from Atari DQN (Minh et al, 2013). 
Each observation from the simulator is a 480x640x3 image.

First I greyscale the image to 480x640 image.
<figure>
    <img src="{{site.baseurl}}/assets/images/grad/gray.png" alt="maes" width="400" height="auto" style="transform:rotate(180deg);">
    <figcaption>Grey scale image</figcaption>
</figure>
Then I downsample the image to 84x84.
<figure>
    <img src="{{site.baseurl}}/assets/images/grad/downsample.png" alt="maes" width="400" height="auto" style="transform:rotate(180deg);">
    <figcaption>Downsample image</figcaption>
</figure>

I also implemented frame skipping and frame stacking.
For frame skipping, I retained every fourth observation and discarded the others.
From the retained frames, I stacked four consecutive observations to construct the final state, resulting in a 4×84×84 tensor.
For better explanation, see the this blog [post](https://danieltakeshi.github.io/2016/11/25/frame-skipping-and-preprocessing-for-deep-q-networks-on-atari-2600-games/).

See my implementation [here](https://github.com/TruongGiangPham-1/CMPUT503FinalProject/blob/b2ad23cd2b8f44ad6b1dbe2e28fb0e3118c635f7/utils.py#L22)

## Results
### 1. Straight Road

<p align="center">
  <img src="{{site.baseurl}}/assets/gifs/503Grad/video_map_straight_road_runlabel_1_evalstep_10_1744507147.gif" width="150"/>
  <img src="{{site.baseurl}}/assets/gifs/503Grad/video_map_straight_road_runlabel_1_evalstep_50_1744508651.gif" width="150"/>
  <img src="{{site.baseurl}}/assets/gifs/503Grad/video_map_straight_road_runlabel_1_evalstep_200_1744514144.gif" width="150"/>
  <img src="{{site.baseurl}}/assets/gifs/503Grad/video_map_straight_road_runlabel_1_evalstep_480_1744524288.gif" width="150"/>
  <figcaption>From Left to right: evaluation episodes after 10th, 50, 200th, 480th update iterations</figcaption>
</p>

<p align="center">
    <img src="{{site.baseurl}}/assets/images/grad/PPO_straight_road_returns.png" alt="maes" width="500" >
    <img src="{{site.baseurl}}/assets/images/grad/PPO_straight_road_duration.png" alt="maes" width="500" >
</p>

I averaged the results across three random seeds—each shown as a grey curve. The red curve represents the running average.

There is no learning going on as expected.
Some of the reason why is mentioned in the limitation.

### 2. Small Loop
<p align="center">
  <img src="{{site.baseurl}}/assets/gifs/503Grad/video_small_loop_evalstep_10.gif" width="250"/>
  <img src="{{site.baseurl}}/assets/gifs/503Grad/video_small_loop_evalstep_50.gif" width="250"/>
  <img src="{{site.baseurl}}/assets/gifs/503Grad/video_small_loop_evalstep_200.gif" width="250"/>
  <img src="{{site.baseurl}}/assets/gifs/503Grad/video_small_loop_evalstep_480.gif" width="250"/>
  <figcaption>From Left to right: evaluation episodes after 10th, 50, 200th, 480th update iterations</figcaption>
</p>

<p align="center">
    <img src="{{site.baseurl}}/assets/images/grad/PPO_small_loop_returns.png" alt="maes" width="500" >
    <img src="{{site.baseurl}}/assets/images/grad/PPO_small_loop_duration.png" alt="maes" width="500" >
</p>

I averaged the results across three random seeds—each shown as a grey curve. The red curve represents the running average.

The duration curve reflects the length of each evaluation episode before termination or truncation. An episode is terminated if the bot goes out of bounds, and it is truncated if the maximum episode length is exceeded. As mentioned earlier, the maximum episode length is set to 10,000 steps.

In the small loop environment, the policy converged to a behavior where the bot spins in circles until it eventually falls off the track.

## Discussion
Both the small_loop agent and the straight road agent's  episode length improves.
This can be attributed to them learning to  stay on the road longer by oscillating
back and forth as seen in the video.

In both experimeent, learning is almost non-existence. 
The undiscounted sum of rewards goes down during the course of training.

## Limitations and Challenges
- The official gym-duckietown repository is really out of date and was challenging
to install. 
Fortunately, there exist an open [pull request](https://github.com/MasWag/gym-duckietown/tree/daffy) with fixes.
- My setup is missing the metric to compute the distance the bot traveled. 
This is because I still cannot get the agent to learn. 
- This version of PPO is still far from being optimized for the lane-following task. Most of the customization has been focused on the CNN architecture, image preprocessing, frame skipping, and frame stacking. Due to time constraints, I was unable to experiment with different PPO hyperparameters. In particular, tuning the entropy coefficient might have improved exploration behavior.
- Reward scaling was not applied. I observed that the rewards can be extremely large and highly variable. Given this, it's not surprising that the agent struggled to learn effectively.

## Future Works
Fully tune the PPO algorithm. There are too many hyperparameters in PPO.
I am now curious to try an off-policy, continuous action RL algorithms like TD3.
TD3 works pretty good out of the box in my experience.
TD3 being able to leverage replay buffer makes it more sample efficient too.

Duckietown has alot of maps.


## References that I used
- [Clean RL PPO](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_continuous_action.py)
- [Atari DQN architecture](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn_atari.py)
- I used ChatGPT to fix my grammar

## Citation
Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal Policy Optimization Algorithms. ArXiv, abs/1707.06347. Retrieved from https://api.semanticscholar.org/CorpusID:28695052

Huang, S., Dossa, R. F. J., Ye, C., Braga, J., Chakraborty, D., Mehta, K., & Araújo, J. G. M. (2022). CleanRL: High-quality Single-file Implementations of Deep Reinforcement Learning Algorithms. Journal of Machine Learning Research, 23(274), 1–18. Retrieved from http://jmlr.org/papers/v23/21-1342.html

Mnih, V., Kavukcuoglu, K., Silver, D., Graves, A., Antonoglou, I., Wierstra, D., & Riedmiller, M. A. (2013). Playing Atari with Deep Reinforcement Learning. CoRR, abs/1312.5602. Retrieved from http://arxiv.org/abs/1312.5602