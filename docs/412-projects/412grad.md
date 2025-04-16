---
layout: page
title: CMPUT 503 grad project
permalink: 412-projects/grad/
date: 2025-02-12 10:33:00 -0000
---
Team member: Truong-Giang Pham

## Introduction
Reinforcement learning methods have been used to drive wheeled robots.
This grad project's purpose is to have hands on experience with the capability
of Proximal Policy Optimzation (PPO) algorithm on learning and performing
lane following tasks.
The initial goal was to compare with PID controller implementation, but this part
is omitted in this project.

In this project, I worked with Gym-Duckietown, a duckietown simulator
writen in openai's gym.
This simulator allows full customization and simulations of the entire duckietown
universe. 
Using simulator not only alleviates problems that comes with real robot, 
but it also enables massive generation of observations for training RL.


TLDR: I planned to do a comparative study of PPO agent and PID controller agent
in gym-duckietown. But I only manage to have a PPO agent so far. 
Therefore, there is no comparative study here. Only a (bad) showcase of a 
PPO agent in gym-duckietown!

## Method
### 1. Environment
Gym-Duckietown is a fast simulator that follows gymnasium convention.
At every time step t, the environment takes in an action [velocity, omega]. 
The velocity is the velocity of the duckiebot, and omega is the turning angle.
The velocity is clipped between -1 and 1, and omega is clipped be
Then, the simulator outputs a 480x640x3 pixel observation after taking that action.
Each action will move the bot in the simulation by a small increment.
I can't quantify how much the bot moves per timestep, but we can see the visuals.

The simulator truncates when we reach 500000 time step, or it terminates when the 
bot goes out of bound.

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
| mini batch_size  | 2048  32 |
| $$\gamma$$  |   0.99 |
| GAE $$\lambda$$  |   0.95 |
| update epoch  | 10 |
| clip_coef  | 0.2 |
| entropy_coef  | 0.1 |
| entropy_coef  | 0.5 |

The exploration is just done using the entropy term.

### 3. Image Pre-Processing
I mimic image processing from Atari DQN.
Each observation from the simulator is a 480x640x3 image.

First I greyscale the image to 480x640 image.
<figure>
    <img src="{{site.baseurl}}/assets/images/grad/gray.png" alt="maes" width="100%" height="auto" style="transform:rotate(180deg);">
    <figcaption>Grey scale image</figcaption>
</figure>
Then I downsample the image to 84x84.
<figure>
    <img src="{{site.baseurl}}/assets/images/grad/downsample.png" alt="maes" width="100%" height="auto" style="transform:rotate(180deg);">
    <figcaption>Downsample image</figcaption>
</figure>

I also implemented frame skipping + frame stacking. 
For frame skipping, I keep every fourth observation and skip the rest. 
Amongst the non-skipped frames, I stack four of them to form a final
4x84x84 tensor as a state.
For concrete example, see the this blog [post](https://danieltakeshi.github.io/2016/11/25/frame-skipping-and-preprocessing-for-deep-q-networks-on-atari-2600-games/).

## Results
Since there is no PID controller agent who can act as a benchmark, 
I will assume that the benchmark can be computed qualitatively.
### 1. Straight Road

<p align="center">
  <img src="{{site.baseurl}}/assets/gifs/503Grad/video_map_straight_road_runlabel_1_evalstep_10_1744507147.gif" width="150"/>
  <img src="{{site.baseurl}}/assets/gifs/503Grad/video_map_straight_road_runlabel_1_evalstep_50_1744508651.gif" width="150"/>
  <img src="{{site.baseurl}}/assets/gifs/503Grad/video_map_straight_road_runlabel_1_evalstep_200_1744514144.gif" width="150"/>
  <img src="{{site.baseurl}}/assets/gifs/503Grad/video_map_straight_road_runlabel_1_evalstep_480_1744524288.gif" width="150"/>
  <figcaption>From Left to right: evaluation frams at 10th, 50, 200th, 480th update iterations</figcaption>
</p>
### 2. Small Loop
This one surprised me. I realized that the agent learnt

## Limitations and Challenges
- The official gym-duckietown repository is really out of date and was challenging
to install. 
Fortunately, there exist an open [pull request](https://github.com/MasWag/gym-duckietown/tree/daffy) with fixes.
- My setup is missing the metric to compute the distance the bot traveled. 
This is because I still cannot get the agent to learn. 
As a result, the discussion is conducted based on the videos genereated.
- This version of PPO is still far from tailored toward this task.
All the customization is saturated in the CNN architecture, image preprocessing, frame skipping, and frame
stacking functionalities.
Due to the lack of time, I was not able to experient with different hyperparameters of PPO.
Specifically, there could be benefits if entropy coefficients is tuned for 
explorations.
- We did not scale the reward. I noticed that the reward can be massive and have high
variance. 
I am not surprised that the agent was not able to learn well. 


## Future Works


There are too many hyperparameters in PPO.
I am now curious to run an off-policy, continuous action RL algorithms like TD3.
TD3 works pretty good out of the box in my experience.
TD3 being able to leverage replay buffer makes it more sample efficient too.



## References that I used
- [Clean RL PPO](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_continuous_action.py)
- [Atari DQN architecture](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn_atari.py)

## Citation
Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal Policy Optimization Algorithms. ArXiv, abs/1707.06347. Retrieved from https://api.semanticscholar.org/CorpusID:28695052

Huang, S., Dossa, R. F. J., Ye, C., Braga, J., Chakraborty, D., Mehta, K., & Araújo, J. G. M. (2022). CleanRL: High-quality Single-file Implementations of Deep Reinforcement Learning Algorithms. Journal of Machine Learning Research, 23(274), 1–18. Retrieved from http://jmlr.org/papers/v23/21-1342.html
