import numpy as np
from collections import deque
import torch
import sys


def ddpg_train(
    agent,
    env,
    brain_name,
    num_agents,
    actor_model_pth,
    critic_model_pth,
    n_episodes=1000,
    max_steps=1000,
    print_every=20,
):

    # Keep track of scores
    scores = []
    scores_window_100 = deque(maxlen=100)

    for episode in range(1, n_episodes + 1):

        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        score = np.zeros(num_agents)
        # Reset the noise in the agents
        agent.reset()

        for t in range(max_steps):
            actions = agent.act(states)
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done
            score += rewards
            agent.step(states, actions, rewards, next_states, dones)

            states = next_states
            if np.any(dones):
                break

        scores_window_100.append(np.mean(score))
        scores.append(np.mean(score))

        if episode % print_every == 0:
            print(
                "\rEpisode {}\tAverage Score: {:.2f}".format(
                    episode, np.mean(scores_window_100)
                )
            )

        if np.mean(scores_window_100) >= 30.0:
            # Agent has reached target average score. Ending training
            print(
                "\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}".format(
                    episode - 100, np.mean(scores_window_100)
                )
            )
            torch.save(
                agent.actor_local.state_dict(), actor_model_pth + "_solution"
            )  # 'actor_model.pth'
            torch.save(
                agent.critic_local.state_dict(), critic_model_pth + "_solution"
            )  #
            break

    return scores


def ddpg_present(agent, env, brain_name, num_agents, actor_model_pth, critic_model_pth):

    # load weights
    agent.actor_local.load_state_dict(torch.load(actor_model_pth))
    agent.actor_local.eval()
    agent.critic_local.load_state_dict(torch.load(critic_model_pth))
    agent.critic_local.eval()

    env_info = env.reset(train_mode=False)[brain_name]  # reset the environment
    states = env_info.vector_observations  # get the current state
    score = np.zeros(num_agents)

    while True:
        actions = agent.act(states)
        env_info = env.step(actions)[brain_name]
        next_states = env_info.vector_observations
        rewards = env_info.rewards
        dones = env_info.local_done
        score += rewards
        states = next_states
        if np.any(dones):
            break
    print("Mean Score (for all the agents): {}".format(np.mean(score)))
    print("Score for individual agents:")
    print(score)
