# Deep Reinforcement Learning Nanodegree - Udacity
This repository contains programming assigments and projects for Udacity's Deep RL Nanodegree program.

## 1. Setup and usage
We would use dockerized env for execution the project files. The requirements stemming from [udacitys
 repo](https://github.com/udacity/deep-reinforcement-learning/blob/master/python/requirements.txt) are 
located [here](./Docker/requirements.txt)

Just update the requirements if necessary, but afterward you should also rebuild the docker-image 
with `docker-compose build --force-rm`. 

After checking out just run `docker-compose up -d jupyter` 
and the jupyter notebook server should be accessible on your localhost on port 8888. If necessary, 
you can modify the [docker-compose.yml](docker-compose.yml)

### 1.1. GPU

If You are using a client with nvidia GPU, there is also a separate dockerfile and service for that.
To use it run `docker-compose up -d jupyter-gpu`  
