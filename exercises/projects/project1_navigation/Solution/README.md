[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# Project 1: Navigation - Solution

### Introduction

This folder contains the solution for the first project in Udacitys Deep Reinforcement Learning Nanodegree program.
The aim of the project was to train an agent to navigate through an environment by avoiding blue and collecting yellow
bananas. The agents gets a reward of +1 for collecting a yellow banana and -1 for collecting a blue one. 

The state space is 37 dimensional, which contains the agent's velocity and the ray based perception of objects around
the agent's forward direction. 

Four discrete actions are available: 
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task can be considered solved when our the agent achieves an average score of 13 over 100 consecutive 
periods.  

![Trained Agent][image1]

###1. Getting started

The Unity based environment is provided by Udacity, you can get the enviroment matching your needs form the links below:

 - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

After downloading the environment please refer to the containing folder in the [notebook](../Navigation.ipynb)

I used for the project a GCP instance with GPU with dockerized environment, thus you can find 
[here](../../../../Docker) the Dockerfiles ([jupyter-CPU](../../../../Docker/Dockerfile) and 
[jupyter-GPU](../../../../Docker/Dockerfile-GPU) and [debug](../../../../Docker/Dockerfile-debug)
the project dependencies in the [`requirements.txt`](../../../../Docker/requirements.txt) file. There is also a 
docker-compose [file](../../../../docker-compose.yml) in the repo for setting up the services. 
Later I will also add the necessary scripts for setting up the instance (e.g. installing CUDA, nvidia-docker, etc.) 
to ensure the best reproducibility. I used the 'headless' version of the environment.   

Since the debugging in jupyter notebooks are rather cumbersome I also set up a 'debug' service in the docker-compose 
file, which is based on a more lightweight docker image (see the dockerfile). Through the experimentation is used 
Pycharm where is set up this docker-compose service as remote interpreter for the project, thus I could debug the 
underlying code through the debugger. After I was sure I had a functioning piece of code, I deployed it on the GCP 
instance and trained it through the jupyter notebook.


###2. Poject-Solution

#### 2.1 Deep Q-Network.
As a baseline solution I implemented a DQN-agent as in the `LunarLander_v2` [exercise](../../../DQN/), 
since the same solution could also applied here with small modifications. The underlying idea for the DQN-Agent you can 
find in DeepMind's 2015 [paper](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)

For the solution I implemented a rather wide 5 layer deep NN. However there wasn't any optimization step through the 
process, fortunately the base architecture seemed working here. 

To execute the solution [notebook](Navigation_solution.ipynb), please place the required folder of the Unity environment 
to the parent folder or modify the reference in the notebook cell accordingly. 

#### 2.1.1. Deep Q-network results
I implemented a rather simple 5 layer deep FC network with [512-256-128-64-32] neurons in the layers. 
For the activations I choose to use Scaled Exponential Linear Units (SELUs) you can find additional information about 
them in [this](https://arxiv.org/pdf/1706.02515.pdf) paper. The solution is based on the DQN exercise, 
which you can find [here](https://github.com/udacity/deep-reinforcement-learning/tree/master/dqn). 
The agent is defined in the `nav_agent.py` [file](./nav_agent.py), where we use two separate networks with the same 
architecture. The target Q-networks weights are updates less often than the local Q-networks weights to increase the 
speed of convergence. The approach is called fixed Q-network. 

The following parameters are defined for the agent:
```BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64  # minibatch size
GAMMA = 0.99  # discount factor
TAU = 1e-3  # for soft update of target parameters
LR = 5e-4  # learning rate
UPDATE_EVERY = 4  # how often to update the network
```
Buffer size defines how many past experiences should the agent hold (memory). As the agent interacts with the 
environment we add experience tuples to the buffer, which are containing the information about the current state, 
the current action, the reward for the current action and the next state and a binary variable indicating if the episode 
is finished. Since the records in the memory are added sequentially they are highly correlated, thus to 'break' this 
correlation we take a random sample from it. 

The `BATCH_SIZE` parameter is the size of the random sample we take from the memory for the learning. 
`GAMMA` is the discount factor, which is the weight for the next Q value.
`TAU` is 'persistence' parameter for the update of the target Q-network weights with the local Q-networks weigths. 
`LR` is the learning rate for the optimizer (`Adam` in our case).
`UPDATE_EVERY` how often should the agent learn (take sample fo buffer and update the weights).
 
 
 With the current approach the agent solved the task in ... steps. 
 ![image2](dqn_agent.png)

#### 2.1.2 Future work

I plan to add different approaches to the solution to be able compare the results. First I plan to implement learning 
from pixels, afterwards would like to go through the variuos improvements of DQN: 

- **Double DQN**
- **Dueling DQN**
- **Prioritized Experience Replay**
- **A3C**
- **Distributinal DQN**
- **Rainbow**


#### 2.2 Deep Q-Network - Pixels

In progress, following soon...:)



