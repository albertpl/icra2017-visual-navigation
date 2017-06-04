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
from gail.discriminator import Discriminator
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
            episode_length += 1
            # ad-hoc reward for navigation
            reward = 10.0 if self.env.terminal else -0.01
            rewards.append(reward)
            terminal = True if episode_length >= self.config.max_steps_per_e else self.env.terminal
            if terminal:
                break
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
        traj_lens = trajs.r.lengths
        t = [self.env.s_target] * n_total
        rewards_stack = self.local_network.d.run_reward(session, self.scene_scope,
                                                      trajs.obs.stacked, t, trajs.actions.stacked)
        assert rewards_stack.shape == (trajs.obs.stacked.shape[0],)
        # convert back to jagged array
        r = RaggedArray(rewards_stack, lengths=trajs.r.lengths)
        B, maxT = len(traj_lens), traj_lens.max()

        # Compute Q values
        q, rewards_B_T = rl.compute_qvals(r, self.config.gamma)
        q_B_T = q.padded(fill=np.nan)
        assert q_B_T.shape == (B, maxT)  # q values, padded with nans at the end

        # Time-dependent baseline that cheats on the current batch
        simplev_B_T = np.tile(np.nanmean(q_B_T, axis=0, keepdims=True), (B, 1))
        assert simplev_B_T.shape == (B, maxT)
        simplev = RaggedArray([simplev_B_T[i, :l] for i, l in enumerate(traj_lens)])

        # State-dependent baseline (value function)
        v_stacked = self.local_network.g.run_value(session, trajs.obs.stacked, t, self.scene_scope)
        assert v_stacked.ndim == 1
        v = RaggedArray(v_stacked, lengths=traj_lens)

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
        adv = RaggedArray([adv_B_T[i, :l] for i, l in enumerate(traj_lens)])
        assert np.allclose(adv.padded(fill=0), adv_B_T)
        return adv, q, vfunc_r2, simplev_r2

    def update_policy(self, session, trajs, adv, t):
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
        # TODO: DEBUG
        logging.info("adv norm =%.2f obj_grad norm=%.2f" % (np.sum(adv.stacked ** 2), float(np.sum(obj_grads**2))))
        if False:
            theta += obj_grads
        else:
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
                                                      trajs.actions.stacked,
                                                      trajs.a_dists.stacked,
                                                      adv.stacked,
                                                      self.scene_scope)
        logging.info("update_policy: obj=%(obj)f kl=%(kl)f" % locals())
        return obj, kl

    def update_value(self, session, trajs, q, writer, t):
        loss = self.local_network.g.step_value(session,
                                               trajs.obs.stacked,
                                               t,
                                               q.stacked,
                                               writer,
                                               self.scene_scope)
        logging.info("update_value: loss=%(loss)f" % locals())
        return loss

    def update_discriminator(self, session, trajs_a, trajs_e, writer, t_a):
        n_total_e = len(trajs_e.obs.stacked)
        t_e = [self.env.s_target] * n_total_e
        loss_d = self.local_network.d.step_d(
            session, self.scene_scope,
            trajs_a.obs.stacked, t_a, trajs_a.actions.stacked,
            trajs_e.obs.stacked, t_e, trajs_e.actions.stacked,
            writer=writer)
        rewards_a = float(np.mean(self.local_network.d.run_reward(
            session, self.scene_scope, trajs_a.obs.stacked, t_a, trajs_a.actions.stacked)))
        rewards_e = float(np.mean(self.local_network.d.run_reward(
            session, self.scene_scope, trajs_e.obs.stacked, t_e, trajs_e.actions.stacked)))
        logging.info("loss_d=%(loss_d)f rewards_a=%(rewards_a)f rewards_e=%(rewards_e)f" % locals())
        return loss_d, rewards_a, rewards_e

    def process(self, session, global_iter, writer):
        obj, loss_v = float('-inf'), float('inf')
        loss_d, rewards_e, rewards_a = float('inf'), float('-inf'), float('-inf')
        # draw experience with current policy and expert policy
        logging.debug("sampling ...")
        trajs_a, trajs_e = self.sample_trajs(session)
        n_total_a, step_a = len(trajs_a.obs.stacked), float(np.mean(trajs_a.obs.lengths))
        n_total_e, step_e = len(trajs_e.obs.stacked), float(np.mean(trajs_e.obs.lengths))
        logging.info("sampled agents %(n_total_a)d pairs (step=%(step_a)f) and "
                     "experts %(n_total_e)d pairs (step=%(step_e)f)" % locals())
        n_total = len(trajs_a.obs.stacked)
        t = [self.env.s_target] * n_total

        # update reward function (discriminator)
        d_cycle = 0
        for _ in range(d_cycle):
            loss_d, rewards_a, rewards_e = self.update_discriminator(
                session, trajs_a, trajs_e, writer, t)

        # update generator network
        g_cycle = 1
        for _ in range(g_cycle):
            # compute Q out of discriminator's reward function and then V out of generator.
            # then advantage = Q - V
            logging.debug("computing advantage")
            adv, q, vfunc_r2, simplev_r2 = self.compute_advantage(session, trajs_a)

            # update policy network via TRPO
            obj, kl = self.update_policy(session, trajs_a, adv, t)

            # update value network via MSE
            #loss_v = self.update_value(session, trajs_a, q, writer, t)

        # add summaries
        summary_dicts = {
            "stats/" + self.scene_scope + "-steps_agent": step_a,
            "stats/" + self.scene_scope + "-steps_expert": step_e,
            "stats/" + self.scene_scope + "-sur_obj": obj,
            "stats/" + self.scene_scope + "-sur_obj": obj,
            "stats/" + self.scene_scope + "-loss_value": loss_v,
            "stats/" + self.scene_scope + "-loss_d": loss_d,
            "stats/" + self.scene_scope + "-rewards_a": rewards_a,
            "stats/" + self.scene_scope + "-rewards_e": rewards_e,
        }
        nn.add_summary(writer, summary_dicts, global_step=global_iter)
        # treat global_t as iteration and increase by one
        return 1

    def evaluate(self, sess, n_episodes, expert_agent=False):
        raise NotImplementedError


def test_model():
    config = Configuration()
    config.max_steps_per_e = 50
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
    max_iter = 10
    with tf.Session(config=sess_config) as session:
        session.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter(train_logdir, session.graph)
        for i in range(max_iter):
            optimizer.process(session, i, writer)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    test_model()
