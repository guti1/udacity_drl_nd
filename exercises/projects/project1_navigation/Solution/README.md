[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# Project 1: Navigation

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

### Getting started

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

### Solution 

The report on the results for this projects can be found [here](./report.md) in a separate document.



