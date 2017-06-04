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
from utils import rl
from utils import nn


class DaggerThread(object):
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
        self.scopes = [network_scope, scene_scope, task_scope]
        self.local_network = global_network
        self.env = Environment({
                'scene_name': self.scene_scope,
                'terminal_state_id': int(self.task_scope)
            })
        self.env.reset()
        self.expert = Expert(self.env)
        self.local_t = 0
        self.episode_length = 0
        self.first_iteration = True # first iteration of Dagger
        # training dataset
        self.states = []
        self.actions = []
        self.targets = []

    def train(self, session, writer):
        assert len(self.states) == len(self.actions), "data count of action and state mismatch"
        s = self.states
        a = self.actions
        n_total = len(s)
        assert n_total > 0, "null dataset"
        t = [self.env.s_target] * n_total
        if n_total > self.config.batch_size:
            data = list(zip(s, a))
            np.random.shuffle(data)
            s, a = zip(*data)
        local_t = self.local_t
        scope = self.scene_scope + '/' + self.task_scope
        for epoch in range(self.config.max_epochs):
            train_loss, train_accuracy = self.local_network.run_epoch(session, self.scopes, s, t, a, True, writer)
            global_step = self.local_network.get_global_step()
            logging.info("%(scope)s:t=%(local_t)d "
                         "train_step=%(global_step)d loss=%(train_loss)f acc=%(train_accuracy)f"
                         % locals())
        return

    def process(self, sess, global_t, summary_writer):
        start_local_t = self.local_t
        # draw experience with current policy or expert policy
        terminal = False
        for i in range(self.config.local_t_max):
            if self.first_iteration:
                # use expert policy before any training
                expert_action = action = self.expert.get_next_action()
                expert_lsr_pi = rl.choose_action_label_smooth(self.config, expert_action, self.config.lsr_epsilon)
            else:
                expert_action = self.expert.get_next_action()
                expert_lsr_pi = rl.choose_action_label_smooth(self.config, expert_action, self.config.lsr_epsilon)
                pi_ = self.local_network.run_policy(sess, self.env.s_t, self.env.s_target, self.scopes)
                action = rl.choose_action(pi_)
                logging.debug("action=%(action)d expert_action=%(expert_action)d "
                              "expert_lsr_pi=%(expert_lsr_pi)s pi_=%(pi_)s" % locals())
            self.states.insert(0, self.env.s_t)
            self.actions.insert(0, expert_lsr_pi)
            self.env.step(action)
            self.env.update()
            terminal = True if self.episode_length > self.config.max_steps_per_e else self.env.terminal
            self.episode_length += 1
            self.local_t += 1
            if terminal:
                logging.info(
                    "[episode end] time %d | thread #%d | scene %s | target #%s expert:%s episode length = %d\n" % (
                        global_t, self.thread_index, self.scene_scope, self.task_scope,
                        "T" if self.first_iteration else "F", self.episode_length))
                summary_values = {
                    "episode_length_input": float(self.episode_length),
                }
                if not self.first_iteration:
                    # record agent's score only
                    nn.add_summary(summary_writer, summary_values,
                                   global_step=self.local_network.get_global_step())
                self.episode_length = 0
                self.env.reset()
                break
        # train policy network with gained labels
        self.train(sess, summary_writer)
        self.first_iteration = False
        return self.local_t - start_local_t

    def evaluate_agent(self, sess, n_episodes, expert_agent):
        ep_lengths = []
        ep_collisions = []
        accuracies = []
        for i in range(n_episodes):
            self.env.reset()
            terminal = False
            step = 0
            n_collision = 0
            while not terminal:
                if expert_agent:
                    action = self.expert.get_next_action()
                else:
                    expert_action = self.expert.get_next_action()
                    pi_ = self.local_network.run_policy(sess, self.env.s_t, self.env.s_target, self.scopes)
                    action = rl.choose_action(pi_)
                    accuracies.append(1.0 if expert_action == action else 0.0)
                    logging.debug("action=%(action)d expert_action=%(expert_action)d pi_=%(pi_)s" % locals())
                self.env.step(action)
                self.env.update()
                terminal = self.env.terminal
                if step > self.config.max_steps_per_e:
                    terminal = True
                    logging.debug("episode %(i)d hits max steps" % locals())
                n_collision += int(self.env.collided)
                step += 1
            logging.debug("episode %(i)d ends with %(step)d steps" % locals())
            ep_lengths.append(step)
            ep_collisions.append(n_collision)
        return np.mean(ep_lengths), np.mean(ep_collisions), np.mean(accuracies)

    def evaluate(self, sess, n_episodes):
        length, collision, acc = self.evaluate_agent(sess, n_episodes, False)
        exp_length, exp_collision, _ = self.evaluate_agent(sess, n_episodes, False)
        return length, acc, exp_length, collision, exp_collision


