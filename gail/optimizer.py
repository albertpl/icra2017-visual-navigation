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
            a, a_dist = self.local_network.g.run_policy(session, [s], [t], self.scene_scope)
            return a[0], a_dist[0]
        trajs_e, trajs_a = [], []
        for _ in range(self.config.min_traj_per_train):
            trajs_a.append(self.sample_one_traj(get_act_fn_a))
            trajs_e.append(self.sample_one_traj(get_act_fn_e))
        return TrajBatch.from_trajs(trajs_a), TrajBatch.from_trajs(trajs_e)

    def compute_advantage(self, session, trajs):
        n_total = len(trajs.obs.stacked)
        trajlengths = trajs.r.lengths
        t = [self.env.s_target] * n_total
        rewards_stack = self.local_network.run_reward(session, self.scene_scope,
                                                      trajs.obs.stacked, t, trajs.actions.stacked)
        assert rewards_stack.shape == (trajs.obs.stacked.shape[0],)
        # convert back to jagged array
        r = RaggedArray(rewards_stack, lengths=trajs.r.lengths)

        B, maxT = len(trajlengths), trajlengths.max()

        # Compute Q values
        q, rewards_B_T = rl.compute_qvals(r, self.config.gamma)
        q_B_T = q.padded(fill=np.nan)
        assert q_B_T.shape == (B, maxT)  # q values, padded with nans at the end

        # Time-dependent baseline that cheats on the current batch
        simplev_B_T = np.tile(np.nanmean(q_B_T, axis=0, keepdims=True), (B, 1))
        assert simplev_B_T.shape == (B, maxT)
        simplev = RaggedArray([simplev_B_T[i, :l] for i, l in enumerate(trajlengths)])

        # State-dependent baseline (value function)
        v_stacked = self.local_network.g.run_value(session, trajs.obs.stacked, t, self.scene_scope)
        assert v_stacked.ndim == 1
        v = RaggedArray(v_stacked, lengths=trajlengths)

        # Compare squared loss of value function to that of the time-dependent value function
        constfunc_prediction_loss = np.var(q.stacked)
        simplev_prediction_loss = np.var(q.stacked - simplev.stacked)  # ((q.stacked-simplev.stacked)**2).mean()
        simplev_r2 = 1. - simplev_prediction_loss / (constfunc_prediction_loss + 1e-8)
        vfunc_prediction_loss = np.var(q.stacked - v_stacked)  # ((q.stacked-v_stacked)**2).mean()
        vfunc_r2 = 1. - vfunc_prediction_loss / (constfunc_prediction_loss + 1e-8)

        # Compute advantage -- GAE(gamma, lam) estimator
        v_B_T = v.padded(fill=0.)
        # append 0 to the right
        v_B_Tp1 = np.concatenate([v_B_T, np.zeros((B, 1))], axis=1)
        assert v_B_Tp1.shape == (B, maxT + 1)
        delta_B_T = rewards_B_T + self.config.gamma * v_B_Tp1[:, 1:] - v_B_Tp1[:, :-1]
        adv_B_T = rl.discount(delta_B_T, self.config.gamma * self.config.lam)
        assert adv_B_T.shape == (B, maxT)
        adv = RaggedArray([adv_B_T[i, :l] for i, l in enumerate(trajlengths)])
        assert np.allclose(adv.padded(fill=0), adv_B_T)
        return adv, q, vfunc_r2, simplev_r2

    def ng_step(self, session, trajs, adv):
        n_total = len(trajs.obs.stacked)
        trajlengths = trajs.r.lengths
        t = [self.env.s_target] * n_total

        def hvp(v):
            return self.local_network.g.run_hvp(session,
                                                trajs.obs.stacked, t,
                                                trajs.actions.stacked, trajs.a_dists.stacked,
                                                v,
                                                self.scene_scope)
        theta = self.local_network.g.get_vars(session, self.scene_scope)

        # compute grads of obj
        _, obj_grads, _ = self.local_network.g.run_sur_obj_kl_with_grads(session,
                                                            trajs.obs.stacked, t,
                                                            trajs.actions.stacked, trajs.a_dists.stacked,
                                                            adv.stacked,
                                                            self.scene_scope)
        # compute hessian with CG
        step_dir = rl.conjugate_gradient(hvp, -obj_grads)
        shs = .5 * step_dir.dot(hvp(step_dir))
        assert shs > 0

        lm = np.sqrt(shs / self.config.policy_max_kl)
        full_step = step_dir / lm
        neg_gdot_step_dir = -obj_grads.dot(step_dir)

        def compute_obj(th):
            self.local_network.g.set_vars(session, th, self.scene_scope)
            return self.local_network.g.run_sur_obj_kl(
                session,
                trajs.obs.stacked, t,
                trajs.actions.stacked, trajs.a_dists.stacked,
                adv.stacked,
                self.scene_scope
            )[0]
        theta = rl.linesearch(compute_obj, theta, full_step, neg_gdot_step_dir/lm)
        self.local_network.g.set_vars(session, theta, self.scene_scope)
        obj, kl = self.local_network.g.run_sur_obj_kl(session,
                                                    trajs.obs.stacked, t,
                                                    trajs.actions.stacked, trajs.a_dists.stacked,
                                                    adv.stacked,
                                                    self.scene_scope)
        logging.info("after ng_step: obj=%(obj)f kl=%(kl)f" % locals())

    def process(self, session, global_t, summary_writer):
        # draw experience with current policy and expert policy
        logging.info("sampling ...")
        trajs_a, trajs_e = self.sample_trajs(session)

        # compute Q out of discriminator's reward function and then V out of generator.
        # then advantage = Q - V
        logging.info("computing advantage")
        adv, q, vfunc_r2, simplev_r2 = self.compute_advantage(session, trajs_a)
        self.ng_step(session, trajs_a, adv)

    def evaluate(self, sess, n_episodes, expert_agent=False):
        raise NotImplementedError


def test_model():
    config = Configuration()
    config.max_steps_per_e = 10
    scene_scopes = ('bathroom_02', 'bedroom_04', 'kitchen_02', 'living_room_08')

    train_logdir = 'logdir'
    discriminator = Discriminator(config, scene_scopes=scene_scopes)
    generator = Generator(config, scene_scopes=scene_scopes)
    model = Network(config, generator, discriminator, scene_scopes=scene_scopes)

    optimizer = GailThread(config, model, 1,
                           scene_scope='bathroom_02',
                           task_scope='1')
    sess_config = tf.ConfigProto(log_device_placement=False,
                                 allow_soft_placement=True)
    max_iter = 100
    with tf.Session(config=sess_config) as session:
        session.run(tf.global_variables_initializer())
        summary_writer = tf.summary.FileWriter(train_logdir, session.graph)
        for _ in range(max_iter):
            optimizer.process(session, 0, summary_writer)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    test_model()
