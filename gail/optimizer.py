# -*- coding: utf-8 -*-
import tensorflow as tf
import logging
import math
import numpy as np
import os
import random
import time
import sys

from scene_loader import THORDiscreteEnvironment as Environment
from policy_network import PolicyNetwork
from expert import Expert
from config import Configuration
from gail.discriminator import  Discriminator
from gail.network import Generator, Network
from gail.trajectory import *
from utils import rl, nn


class GailThread(object):
    def __init__(self,
                 config,
                 global_network,
                 thread_index,
                 network_scope="network",
                 scene_scope="scene",
                 task_scope="task"):
        self.thread_index = thread_index
        self.config = config
        self.network_scope = network_scope
        self.scene_scope = scene_scope
        self.task_scope = task_scope
        self.scopes = (network_scope, scene_scope, task_scope)
        self.local_network = global_network
        self.env = Environment({
                'scene_name': self.scene_scope,
                'terminal_state_id': int(self.task_scope)
            })
        self.env.reset()
        self.expert = Expert(self.env)
        self.local_t = 0
        self.first_iteration = True # first iteration of Dagger
        # training dataset
        self.states = []
        self.actions = []
        self.targets = []

    def sample_one_traj(self, policy_fn):
        terminal = False
        episode_length = 0
        states, actions, a_dists, rewards = [], [], [], []
        self.env.reset()
        while not terminal:
            a, a_dist = policy_fn(self.env.s_t, self.env.s_target)
            states.append(self.env.s_t)
            actions.append(a)
            a_dists.append(a_dist)
            self.env.step(a)
            self.env.update()
            # ad-hoc reward for navigation
            reward = 10.0 if self.env.terminal else -0.01
            rewards.append(reward)
            terminal = True if episode_length > self.config.max_steps_per_e else self.env.terminal
            if terminal:
                break
            episode_length += 1
        return Trajectory(np.array(states), np.array(a_dists), np.array(actions), np.array(rewards))

    def sample_trajs(self, session):
        def get_act_fn_e(s, t):
                a = self.expert.get_next_action()
                a_dist = rl.choose_action_label_smooth(self.config, a, self.config.lsr_epsilon)
                return a, a_dist

        def get_act_fn_a(s, t):
                a, a_dist = self.local_network.g.run_policy(session, [s], [t], self.scopes)
                return a[0], a_dist[0]
        trajs_e, trajs_a = [], []
        for _ in range(self.config.min_traj_per_train):
            trajs_a.append(self.sample_one_traj(get_act_fn_a))
            trajs_e.append(self.sample_one_traj(get_act_fn_e))
        return TrajBatch.from_trajs(trajs_a), TrajBatch.from_trajs(trajs_e)

    def process(self, session, global_t, summary_writer):
        # draw experience with current policy and expert policy
        trajs_a, trajs_e = self.sample_trajs(session)
        return self.config.min_traj_per_train

    def evaluate(self, sess, n_episodes, expert_agent=False):
        raise NotImplementedError


def test_model():
    config = Configuration()
    config.max_steps_per_e = 10
    network_scope = 'network_scope'
    scene_scopes = ('bathroom_02', 'bedroom_04', 'kitchen_02', 'living_room_08')

    train_logdir = 'logdir'
    discriminator = Discriminator(config, network_scope=network_scope, scene_scopes=scene_scopes)
    generator = Generator(config, network_scope=network_scope, scene_scopes=scene_scopes)
    model = Network(config, generator, discriminator, network_scope=network_scope, scene_scopes=scene_scopes)

    optimizer = GailThread(config, model, 1,
                           network_scope=network_scope,
                           scene_scope='bathroom_02',
                           task_scope='1')
    sess_config = tf.ConfigProto(log_device_placement=False,
                                 allow_soft_placement=True)
    batch_size = config.batch_size
    max_iter = 1
    with tf.Session(config=sess_config) as session:
        session.run(tf.global_variables_initializer())
        summary_writer = tf.summary.FileWriter(train_logdir, session.graph)
        for iter in range(max_iter):
            optimizer.process(session, 0, summary_writer)

if __name__ == '__main__':
    test_model()
