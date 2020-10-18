# Project2: Continous control

## Introduction
This folder contains the solution for the second project in Udacity's Deep Reinforcement Learning Nanodegree program. 
The aim of the project is to train an agent, who can move a double-jointed arm to reach the target location. A reward 
of 0.1 is provided for each step when the agent's 'hand' is in the target location. Thus the goal is to  

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step
that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the 
target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular 
velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two 
joints. Every entry in the action vector should be a number between -1 and 1.

There are two options to solve the project:

1. The Unity environment contain a single agent
2. The environment contains 20 identical agents, each with its own copy of the environment.

Accordingly the criteria to solve the environment are the following:

1. The task is episodic, and in order to solve the environment, the agent must get an average score of +30 over 
100 consecutive episodes.
2. In particular, the agents must get an average score of +30 (over 100 consecutive episodes, and over all 
agents). Specifically,

    - After each episode, we add up the rewards that each agent received (without discounting), to get a score for 
    each agent. This yields 20 (potentially different) scores. We then take the average of these 20 scores.
    - This yields an average score for each episode (where the average is over all 20 agents).

The environment is considered solved, when the average (over 100 episodes) of those average scores is at least +30.


## Setup

-  Download the chosen version from the links below:
    - **_Version 1: One (1) Agent_**
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)

    - **_Version 2: Twenty (20) Agents_**
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) 
    if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows 
    operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux_NoVis.zip) (version 1) or [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux_NoVis.zip) 
    (version 2) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent 
    without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should 
    follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)
    , and then download the environment for the **Linux** operating system above._)
    
  Since the setup of the repo is aimed for docker-based development I suggest to choose the headless versions. After 
  downloading the environment please check the usage in the notebook [here](../Continuous_Control.ipynb)
  
- Docker-based execution

    I used for the project a GCP instance with GPU with dockerized environment, thus you can find here the Dockerfiles 
    (jupyter-CPU and jupyter-GPU and debug the project dependencies in the requirements.txt file. There is also a 
    docker-compose file in the repo for setting up the services. Later I will also add the necessary scripts for 
    setting up the instance (e.g. installing CUDA, nvidia-docker, etc.) to ensure the best reproducibility. I used the 
    'headless' version of the environment.

    Since the debugging in jupyter notebooks are rather cumbersome I also set up a 'debug' service in the 
    docker-compose file, which is based on a more lightweight docker image (see the dockerfile). Through the 
    experimentation is used Pycharm where is set up this docker-compose service as remote interpreter for the project, 
    thus I could debug the underlying code through the debugger. After I was sure I had a functioning piece of code, I 
    deployed it on the GCP instance and trained it through the jupyter notebook.

## Solution
The report on the results for this projects can be found [here](./), along with the notebook to reproduce the 
reported results as well.

The usage of the agent and the utils are presented in the [solution notebook](./solution.ipynb). The configuration for 
the experiments along with the plots and the saved network weights are located [here](./experiments).

The report about the experiments with the hyperparameters can be found [here](Report.md).

