# -*- coding: utf-8 -*-
import tensorflow as tf
import logging
import math
import numpy as np
import os
import random
import time
import sys
import scipy.sparse.linalg as ssl

from scene_loader import THORDiscreteEnvironment as Environment
from expert import Expert
from config import Configuration
from imitation.discriminator import Discriminator
from imitation.network import Generator, Network
from imitation.sample import *
from utils.trajectory import *
from utils import rl, nn



class GailThread(object):
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
            _, a_dist = self.local_network.g.run_policy(session, [s], [t], self.scene_scope)
            a = rl.choose_action(a_dist[0])
            return a, a_dist[0]
        trajs = []
        for _ in range(n_traj):
            trajs.append(sample_one_traj(self.config, self.env, get_act_fn_a))
        return TrajBatch.from_trajs(trajs)

    def compute_advantage(self, session, trajs):
        n_total = len(trajs.obs.stacked)
        traj_lens = trajs.r.lengths
        t = [self.env.s_target] * n_total
        rewards_stack = self.local_network.d.run_reward(session, self.scene_scope,
                                                        trajs.obs.stacked, t, trajs.actions.stacked)
        assert rewards_stack.shape == (trajs.obs.stacked.shape[0],)
        if self.config.policy_ent_reg > 0:
            # add casual entropy as reward augmentation
            ent = self.local_network.g.run_ent(session, trajs.obs.stacked, t, trajs.actions.stacked, self.scene_scope)
            rewards_stack += self.config.policy_ent_reg * ent

        # convert back to jagged array
        r = RaggedArray(rewards_stack, lengths=trajs.r.lengths)
        B, maxT = len(traj_lens), traj_lens.max()

        # Compute Q values
        q, rewards_B_T = rl.compute_qvals(r, self.config.gamma)
        q_B_T = q.padded(fill=np.nan)
        assert q_B_T.shape == (B, maxT)  # q values, padded with nans at the end

        # estimate expected return
        returns = np.mean(np.sum(rewards_B_T, axis=1))

        # State-dependent baseline (value function)
        v_stacked = self.local_network.g.run_value(session, trajs.obs.stacked, t, self.scene_scope)
        assert v_stacked.ndim == 1
        v = RaggedArray(v_stacked, lengths=traj_lens)

        # Compute advantage -- GAE(gamma, lam) estimator
        v_B_T = v.padded(fill=0.)
        # append 0 to the right
        v_B_Tp1 = np.concatenate([v_B_T, np.zeros((B, 1))], axis=1)
        assert v_B_Tp1.shape == (B, maxT + 1)
        delta_B_T = rewards_B_T + self.config.gamma * v_B_Tp1[:, 1:] - v_B_Tp1[:, :-1]
        adv_B_T = rl.discount(delta_B_T, self.config.gamma * self.config.gae_lam)
        assert adv_B_T.shape == (B, maxT)
        adv = RaggedArray([adv_B_T[i, :l] for i, l in enumerate(traj_lens)])
        assert np.allclose(adv.padded(fill=0), adv_B_T)
        return adv, q, returns

    def update_policy(self, session, trajs, adv, t):
        def damped_hvp_func(v):
            hvp = self.local_network.g.run_hvp(session,
                                                trajs.obs.stacked, t,
                                                trajs.actions.stacked, trajs.a_dists.stacked,
                                                v,
                                                self.scene_scope)
            return hvp + self.config.policy_cg_damping * v
        theta = self.local_network.g.get_vars(session, self.scene_scope)

        # compute grads of obj
        obj_old, obj_grads, kl = self.local_network.g.run_sur_obj_kl_with_grads(session,
                                                            trajs.obs.stacked, t,
                                                            trajs.actions.stacked, trajs.a_dists.stacked,
                                                            adv.stacked,
                                                            self.scene_scope)
        # TODO: DEBUG
        obj_grad_norm = np.abs(obj_grads).max()
        if obj_grad_norm < 1e-6:
            adv_norm = np.abs(adv.stacked).max()
            logging.info("adv norm =%.2f obj_grad norm=%.2f" % (adv_norm, obj_grad_norm))
            return obj_old, kl
        else:
            # compute hessian with CG
            def barrier_obj(th):
                self.local_network.g.set_vars(session, th, self.scene_scope)
                obj, kl = self.local_network.g.run_sur_obj_kl(
                    session,
                    trajs.obs.stacked, t,
                    trajs.actions.stacked, trajs.a_dists.stacked,
                    adv.stacked,
                    self.scene_scope
                )
                return np.inf if kl > 2 * self.config.policy_max_kl else -obj
            if False:
                hvpop = ssl.LinearOperator(shape=(theta.shape[0], theta.shape[0]), matvec=damped_hvp_func)
                step, _ = ssl.cg(hvpop, obj_grads, maxiter=10)
            else:
                step = rl.conjugate_gradient(damped_hvp_func, obj_grads)
            full_step = step / np.sqrt(.5 * step.dot(damped_hvp_func(step)) / self.config.policy_max_kl)
            theta, num_bt_steps = rl.btlinesearch(
                f=barrier_obj,
                x0=theta,
                fx0=-obj_old,
                g=-obj_grads,
                dx=full_step,
                accept_ratio=.1, shrink_factor=.5, max_steps=10)
        self.local_network.g.set_vars(session, theta, self.scene_scope)
        obj, kl = self.local_network.g.run_sur_obj_kl(session,
                                                      trajs.obs.stacked, t,
                                                      trajs.actions.stacked,
                                                      trajs.a_dists.stacked,
                                                      adv.stacked,
                                                      self.scene_scope)
        logging.debug("update_policy: obj=%(obj)f obj_old=%(obj_old)f kl=%(kl)f" % locals())
        return obj, kl

    def update_value(self, session, trajs, q, writer, t):
        loss = self.local_network.g.step_value(session,
                                               trajs.obs.stacked,
                                               t,
                                               q.stacked,
                                               writer,
                                               self.scene_scope)
        logging.debug("update_value: loss=%(loss)f" % locals())
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
        logging.debug("loss_d=%(loss_d)f rewards_a=%(rewards_a)f rewards_e=%(rewards_e)f" % locals())
        return loss_d, rewards_a, rewards_e

    def process(self, session, global_t, writer):
        obj, loss_v = float('-inf'), float('inf')
        loss_d, rewards_e, rewards_a = float('inf'), float('-inf'), float('-inf')
        # draw experience with current policy and expert policy
        trajs_a, trajs_e = self.sample_trajs_a(session, self.config.min_traj_per_train), \
                           self.sample_trajs_e(session, self.config.min_traj_per_train)
        n_total_a, step_a = len(trajs_a.obs.stacked), float(np.mean(trajs_a.obs.lengths))
        n_total_e, step_e = len(trajs_e.obs.stacked), float(np.mean(trajs_e.obs.lengths))
        logging.debug("sampled agents %(n_total_a)d pairs (step=%(step_a)f) and "
                      "experts %(n_total_e)d pairs (step=%(step_e)f)" % locals())
        t = [self.env.s_target] * n_total_a

        # update reward function (discriminator)
        d_cycle = self.config.gan_d_cycle
        for _ in range(d_cycle):
            loss_d, rewards_a, rewards_e = self.update_discriminator(
                session, trajs_a, trajs_e, writer, t)

        # update generator network
        # compute Q out of discriminator's reward function and then V out of generator.
        # then advantage = Q - V
        logging.debug("computing advantage")
        adv, q, returns = self.compute_advantage(session, trajs_a)

        # update policy network via TRPO
        for _ in range(self.config.gan_p_cycle):
            obj, kl = self.update_policy(session, trajs_a, adv, t)

        # update value network via MSE
        v_cycle = self.config.gan_v_cycle
        for _ in range(v_cycle):
            loss_v = self.update_value(session, trajs_a, q, writer, t)

        # add summaries
        summary_dicts = {
            "stats/" + self.scene_scope + "-steps_agent": step_a,
            "stats/" + self.scene_scope + "-sur_obj": obj,
            "stats/" + self.scene_scope + "-loss_value": loss_v,
            "stats/" + self.scene_scope + "-loss_d": loss_d,
            "stats/" + self.scene_scope + "-returns": returns,
            "stats/" + self.scene_scope + "-rewards_a": rewards_a,
            "stats/" + self.scene_scope + "-rewards_e": rewards_e,
        }
        nn.add_summary(writer, summary_dicts, global_step=global_t)
        return n_total_a + n_total_e

    def evaluate(self, session, n_episodes):
        trajs_a, trajs_e = self.sample_trajs_a(session, n_episodes), \
                           self.sample_trajs_e(session, n_episodes)
        step_a, step_e = trajs_a.obs.lengths, trajs_e.obs.lengths
        return step_a, 0.0, step_e, 0.0, 0.0


def test_model():
    config = Configuration()
    config.max_steps_per_e = 50
    scene_scopes = ('bathroom_02', 'bedroom_04', 'kitchen_02', 'living_room_08')

    train_logdir = 'logdir' + '/ut_gail'
    discriminator = Discriminator(config, scene_scopes=scene_scopes)
    generator = Generator(config, scene_scopes=scene_scopes)
    model = Network(config, generator, discriminator, scene_scopes=scene_scopes)

    thread = GailThread(config, model, 1,
                           scene_scope='bathroom_02',
                           task_scope='1')
    sess_config = tf.ConfigProto(log_device_placement=False,
                                 allow_soft_placement=True)
    max_iter = 10
    with tf.Session(config=sess_config) as session:
        session.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter(train_logdir, session.graph)
        for i in range(max_iter):
            thread.process(session, i, writer)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    test_model()
