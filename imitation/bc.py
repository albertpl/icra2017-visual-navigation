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
from expert import Expert
from config import Configuration
from imitation.discriminator import Discriminator
from imitation.network import Generator, Network
from imitation.sample import sample_one_traj, sample_target
from utils.trajectory import *
from utils import rl, nn


class BCThread(object):
    def __init__(self,
                 config,
                 global_network,
                 thread_index,
                 network_scope="network",
                 scene_scope="scene",
                 task_scope="task",
                 random_target=False):
        self.thread_index = thread_index
        self.config = config
        self.network_scope = network_scope
        self.scene_scope = scene_scope
        self.task_scope = task_scope
        self.local_network = global_network
        self.env = Environment({
                'scene_name': self.scene_scope,
                'terminal_state_id': int(self.task_scope)
            })
        if random_target:
            new_target = sample_target(self.env)
            logging.info("new_target=%(new_target)d" % locals())
            self.env = Environment({
                'scene_name': self.scene_scope,
                'terminal_state_id': new_target,
            })
        self.env.reset()
        self.expert = Expert(self.env)
        self.local_t = 0
        self.first_iteration = True # first iteration of Dagger
        # training dataset
        self.states = []
        self.actions = []
        self.targets = []

    def sample_trajs_e(self, session, n_traj):
        def get_act_fn_e(s, t):
            a = self.expert.get_next_action()
            a_dist = rl.choose_action_label_smooth(self.config, a, self.config.lsr_epsilon)
            return a, a_dist
        trajs_e = []
        for _ in range(n_traj):
            trajs_e.append(sample_one_traj(self.config, self.env, get_act_fn_e))
        return TrajBatch.from_trajs(trajs_e)

    def sample_trajs_a(self, session, n_traj):
        def get_act_fn_a(s, t):
            a, a_dist = self.local_network.g.run_policy(session, [s], [t], self.scene_scope)
            return a[0], a_dist[0]
        trajs = []
        for _ in range(n_traj):
            trajs.append(sample_one_traj(self.config, self.env, get_act_fn_a))
        return TrajBatch.from_trajs(trajs)

    def process(self, session, global_t, writer):
        # draw experience with expert policy
        logging.debug("sampling ...")
        trajs_e = self.sample_trajs_e(session, self.config.min_traj_per_train)
        n_total_e, step_e = len(trajs_e.obs.stacked), float(np.mean(trajs_e.obs.lengths))
        logging.debug("sampled experts %(n_total_e)d pairs (step=%(step_e)f)" % locals())
        t = [self.env.s_target] * n_total_e

        # only policy network in generator
        acc, loss = self.local_network.g.step_policy(
            session, trajs_e.obs.stacked, t, trajs_e.a_dists.stacked, writer, self.scene_scope)

        # evaluate with agent policy
        trajs = self.sample_trajs_a(session, self.config.num_eval_episodes)
        step_a = float(np.mean(trajs.obs.lengths))
        logging.debug("evaluating agent step=%(step_a)f" % locals())

        # add summaries
        summary_dicts = {
            "stats/" + self.scene_scope + "-steps_agent": step_a,
            "stats/" + self.scene_scope + "-loss_p": loss,
            "stats/" + self.scene_scope + "-acc_p": acc,
        }
        nn.add_summary(writer, summary_dicts, global_step=global_t)
        # treat global_t as iteration and increase by one
        return n_total_e

    def evaluate(self, session, n_episodes):
        trajs_a, trajs_e = self.sample_trajs_a(session, n_episodes), \
                           self.sample_trajs_e(session, n_episodes)
        step_a, step_e = trajs_a.obs.lengths, trajs_e.obs.lengths
        n_total_e = len(trajs_e.obs.stacked)
        t = [self.env.s_target] * n_total_e
        acc, _ = self.local_network.g.eval_policy(
            session, trajs_e.obs.stacked, t, trajs_e.a_dists.stacked, self.scene_scope)
        return step_a, acc, step_e, 0.0, 0.0


def test_model():
    config = Configuration()
    config.max_steps_per_e = 50
    scene_scopes = ('bathroom_02', 'bedroom_04', 'kitchen_02', 'living_room_08')

    train_logdir = 'logdir' + '/ut_bc'
    discriminator = Discriminator(config, scene_scopes=scene_scopes)
    generator = Generator(config, scene_scopes=scene_scopes)
    model = Network(config, generator, discriminator, scene_scopes=scene_scopes)

    optimizer = BCThread(config, model, 1, scene_scope='bathroom_02', task_scope='1')
    sess_config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    max_iter = 10
    with tf.Session(config=sess_config) as session:
        session.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter(train_logdir, session.graph)
        for i in range(max_iter):
            optimizer.process(session, i, writer)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    test_model()
