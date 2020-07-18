# Proximal policy optimization

## REINFORCE vs ppo

The `pong-REINFORCE` notebooks are containing the exercise for implementing the REINFORCE algo to play pong. You can 
see that with the given settings and policy-network architecture the agent needs 1500-2000 episodes to be able to beat
the env. 

 - [pong-reinforce - exercise](./pong-REINFORCE.ipynb)
 - [pong-reinforce - solution](./pong-REINFORCE_sol.ipynb) 
 
 You can find the basic solutions in the accompanied [pong-utils](./pong_utils.py) along with some helper functions as
 in [parallelEnv](./parallelEnv.py) where the parallel execution framework for the game environment is located.
 
 Using PPO for the pong exercise the convergence seems to happening faster (after 500 episodes), however the mean 
 rewards stayed in the negative range. 
 
 Update on `2020.07.18`:  - at least for now the mean reward around 0.  
 
 - [pong-ppo - exercise](./pong-PPO.ipynb)
 - [pong-ppo - solution](./pong-PPO_sol.ipynb)
 
 Maybe some hyper-parameter tuning as epsilon etc. could help. More to come...
