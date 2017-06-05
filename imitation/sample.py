import tensorflow as tf
import logging
from utils.trajectory import *
from utils import rl, nn


def sample_one_traj(config, env, policy_fn):
    terminal = False
    episode_length = 0
    states, actions, a_dists, rewards = [], [], [], []
    env.reset()
    while not terminal:
        a, a_dist = policy_fn(env.s_t, env.s_target)
        states.append(env.s_t)
        actions.append(a)
        a_dists.append(a_dist)
        env.step(a)
        env.update()
        episode_length += 1
        # ad-hoc reward for navigation
        reward = 10.0 if env.terminal else -0.01
        rewards.append(reward)
        terminal = True if episode_length >= config.max_steps_per_e else env.terminal
        if terminal:
            break
    return Trajectory(np.array(states), np.array(a_dists), np.array(actions), np.array(rewards))


