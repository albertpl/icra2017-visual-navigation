# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.contrib import layers
import logging
import math
import numpy as np
from utils import nn, rl
from config import Configuration
from imitation.discriminator import Discriminator

tf.logging.set_verbosity(tf.logging.DEBUG)


# generator
class Generator(object):
    def __init__(self,
                 config,
                 device='/cpu:0',
                 network_scope='generator',
                 scene_scopes=('scene',)):
        self._device = device
        self.config = config
        self.network_scope, self.scene_scopes = network_scope, scene_scopes
        tf.logging.debug('config=%s', self.config)

        self.evals_p, self.summaries_v, self.summaries_p, self.actions = {}, {}, {}, {}
        self.train_vars_p, self.train_vars_v, self.actions_dists, self.values = {}, {}, {}, {}
        self.kl, self.sur_obj, self.hvp, self.get_op, self.set_op = {}, {}, {}, {}, {}
        self.var_vs, self.fc1, self.fc2 = None, None, None
        self.obj_grad, self.kl_grad, self.fc3, self.logits = {}, {}, {}, {}
        self.loss_v, self.loss_p, self.train_v, self.train_p, self.log_p = {}, {}, {}, {}, {}
        with tf.variable_scope(network_scope):
            self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
            self.s = tf.placeholder(tf.float32, [None, 2048, 4], name='observation')
            self.t = tf.placeholder(tf.float32, [None, 2048, 4], name='target')
            self.n = tf.placeholder(tf.int32, [], name='n')   # actual batch size
            self.a_old = tf.placeholder(tf.int32, [None], name='a_old')  # current action from pi_old
            self.a_dist_old = tf.placeholder(tf.float32, [None, 4], name='a_dist_old')  # pi_old(.)
            self.adv = tf.placeholder(tf.float32, [None], name='advantage')
            self.v = tf.placeholder(tf.float32, [None], name='vector')
            self.target_value = tf.placeholder(tf.float32, [None], name='target_value')
            self.is_training = tf.placeholder(tf.bool, name='is_training')
            self.lr = tf.placeholder(tf.float32, [], name='lr')
            self.build_graph()

    def build_graph(self):
        with tf.device(self._device):
            fc1_out = nn.siamese(self.s, self.t, 512, name='fc1')
            self.fc1 = fc1_out
            tf.logging.debug("%s fc1: shape %s", self.network_scope, fc1_out.get_shape())
            # shared fusion layer
            fc2_out = tf.layers.dense(fc1_out, 512, name='fc2', activation=nn.leaky_relu,
                                      kernel_initializer=layers.variance_scaling_initializer())
            self.fc2 = fc2_out
            tf.logging.debug("%s fc2: shape %s", self.network_scope, fc2_out.get_shape())
            shared_variables = tf.trainable_variables()
            tf.logging.debug('shared variables %s ', shared_variables)
            for scene_scope in self.scene_scopes:
                # scene-specific key
                key = rl.get_key([self.network_scope, scene_scope])
                with tf.variable_scope(scene_scope):
                    # scene-specific adaptation layer, disable bn to make it easier for optimizer op dependency
                    x = tf.layers.dense(fc2_out, 512, name='fc3', activation=nn.leaky_relu,
                                        kernel_initializer=layers.variance_scaling_initializer())
                    tf.logging.debug("%s-fc3: shape %s", key, x.get_shape())
                    self.fc3[key] = x
                    x = tf.layers.dense(x, 512, name='fc4', activation=nn.leaky_relu,
                                        kernel_initializer=layers.variance_scaling_initializer())
                    scene_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=key)

                    # policy output layer
                    with tf.variable_scope('policy') as scope:
                        logits = tf.layers.dense(x, self.config.action_size, name='action_logits', activation=None,
                                                 kernel_initializer=layers.variance_scaling_initializer())
                        self.logits[key] = logits
                        self.actions_dists[key] = tf.nn.softmax(logits=logits)
                        tf.logging.debug("%s-action_dist: shape %s", key, self.actions_dists[key].get_shape())
                        self.actions[key] = tf.cast(tf.reshape(
                            tf.argmax(logits, axis=1), (-1,)), tf.int32, name='action_a')
                        tf.logging.debug("%s-action: shape %s", key, self.actions[key].get_shape())
                        policy_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.name)
                        self.train_vars_p[key] = shared_variables + scene_variables + policy_variables
                        tf.logging.debug('scene %s policy variables %s ', key, self.train_vars_p[key])

                    # value output layer
                    with tf.variable_scope('value') as scope:
                        x = tf.layers.dense(self.fc3[key], 256, name='value_fc', activation=nn.leaky_relu,
                                            kernel_initializer=layers.variance_scaling_initializer())
                        values = tf.layers.dense(x, 1, name='value_logits', activation=None,
                                                 kernel_initializer=layers.variance_scaling_initializer())
                        self.values[key] = tf.reshape(values, (-1,))
                        tf.logging.debug("%s-values: shape %s", key, self.values[key].get_shape())
                        value_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.name)
                        self.train_vars_v[key] = value_variables
                        tf.logging.debug('scene %s value variables %s ', key, self.train_vars_v[key])

                    # TRPO ops
                    var_list = self.train_vars_p[key]
                    if self.var_vs is None:
                        total_size = sum([nn.numel(v) for v in var_list])
                        self.var_vs = tf.placeholder(tf.float32, (total_size,), name='var_vs')
                    self.set_op[key] = nn.add_assign_from_flat(var_list, self.var_vs)
                    self.get_op[key] = nn.add_get_from_flat(var_list)
                    # pi(a) pi_old(a)
                    a_dist = self.actions_dists[key]
                    a_dist_old = self.a_dist_old
                    pi = tf.gather_nd(a_dist,
                                      tf.stack((tf.range(self.n), self.a_old), axis=1), name='pi')
                    pi_old = tf.gather_nd(a_dist_old,
                                          tf.stack((tf.range(self.n), self.a_old), axis=1), name='pi_old')
                    # surrogate objective
                    epsilon = 1e-8
                    # minimize obj, i.e. loss
                    self.log_p[key] = tf.log(pi+epsilon)
                    self.sur_obj[key] = tf.reduce_mean(
                        tf.exp(self.log_p[key] - tf.log(pi_old+epsilon)) * self.adv, name='sur_obj')
                    tf.logging.debug("%s-sur_obj: shape %s", key, self.sur_obj[key].get_shape())
                    self.obj_grad[key] = nn.flat_grad(self.sur_obj[key], var_list)
                    # KL divergence
                    kl = tf.reduce_sum(a_dist_old * (tf.log(a_dist_old+epsilon) - tf.log(a_dist+epsilon)), axis=1, name='KL')
                    self.kl[key] = tf.reduce_mean(kl)
                    self.kl_grad[key] = nn.flat_grad(self.kl[key], var_list)
                    # hessian vector product
                    self.hvp[key] = nn.flat_grad(tf.reduce_sum(self.kl_grad[key] * self.v, name='HVP'), var_list)

                # supervised learning for policy
                with tf.name_scope('policy_update/' + scene_scope):
                    self.loss_p[key] = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                        labels=self.a_dist_old,
                        logits=self.logits[key],
                    ))
                    self.summaries_p[key] = tf.summary.scalar('loss_p', self.loss_p[key])
                    optimizer = tf.train.AdamOptimizer(learning_rate=self.lr, name='adam')
                    self.train_p[key] = optimizer.minimize(
                        self.loss_p[key], global_step=self.global_step,
                        name='train_p', var_list=self.train_vars_p[key])
                    a_old = tf.argmax(self.a_dist_old, axis=1)
                    a_pred = tf.argmax(self.logits[key], axis=1)
                    self.evals_p[key] = tf.reduce_mean(tf.cast(tf.equal(a_old, a_pred), tf.float32))

                # supervised learning for value
                with tf.name_scope('value_update/' + scene_scope):
                    self.loss_v[key] =\
                        0.5 * tf.reduce_mean(tf.square(self.target_value - self.values[key]), name='mse')
                    self.summaries_v[key] = tf.summary.scalar('loss_v', self.loss_v[key])
                    optimizer = tf.train.AdamOptimizer(learning_rate=self.lr, name='adam')
                    self.train_v[key] = optimizer.minimize(
                        self.loss_v[key], global_step=self.global_step,
                        name='train_v', var_list=self.train_vars_v[key])
        return

    def run_value(self, session, state, target, scene_scope):
        assert len(state) == len(target)
        key = rl.get_key([self.network_scope, scene_scope])
        v = session.run(self.values[key], feed_dict={
            self.s: state,
            self.t: target,
        })
        return v

    def step_value(self, session, state, target, target_value, writer, scene_scope):
        assert len(state) == len(target) == len(target_value)
        key = rl.get_key([self.network_scope, scene_scope])
        ops = (self.loss_v[key], self.summaries_v[key], self.train_v[key])
        loss, summary, _ = session.run(ops, feed_dict={
            self.s: state,
            self.t: target,
            self.lr: self.config.lr_vn,
            self.target_value: target_value,
        })
        writer.add_summary(summary)
        return loss

    def run_policy(self, session, state, target, scene_scope):
        assert len(state) == len(target)
        key = rl.get_key([self.network_scope, scene_scope])
        return session.run([self.actions[key], self.actions_dists[key]], feed_dict={
            self.s: state,
            self.t: target,
        })

    def step_policy(self, session, state, target, a_dist_target, writer, scene_scope):
        assert len(state) == len(target) == len(a_dist_target)
        key = rl.get_key([self.network_scope, scene_scope])
        ops = (self.evals_p[key], self.loss_p[key], self.summaries_p[key], self.train_p[key])
        acc, loss, summary, _ = session.run(ops, feed_dict={
            self.s: state,
            self.t: target,
            self.a_dist_old: a_dist_target,
            self.lr: self.config.lr,
        })
        writer.add_summary(summary)
        return acc, loss

    def eval_policy(self, session, state, target, a_dist_target, scene_scope):
        assert len(state) == len(target) == len(a_dist_target)
        key = rl.get_key([self.network_scope, scene_scope])
        ops = (self.evals_p[key], self.loss_p[key])
        acc, loss = session.run(ops, feed_dict={
            self.s: state,
            self.t: target,
            self.a_dist_old: a_dist_target,
        })
        return acc, loss

    # TRPO functions
    def run_sur_obj_kl(self, session, state, target, action, a_dist, adv, scene_scope):
        assert len(state) == len(target) == len(action) == len(a_dist)
        key = rl.get_key([self.network_scope, scene_scope])
        ret = session.run([self.sur_obj[key], self.kl[key]], feed_dict={
            self.s: state,
            self.t: target,
            self.a_old: action,
            self.a_dist_old: a_dist,
            self.adv: adv,
            self.n: len(state),
        })
        return ret

    def run_sur_obj_kl_with_grads(self, session, state, target, action, a_dist, adv, scene_scope):
        key = rl.get_key([self.network_scope, scene_scope])
        assert len(state) == len(target) == len(action) == len(a_dist)
        ret = session.run([self.sur_obj[key], self.obj_grad[key], self.kl[key]], feed_dict={
            self.s: state,
            self.t: target,
            self.a_old: action,
            self.a_dist_old: a_dist,
            self.adv: adv,
            self.n: len(state),
        })
        return ret

    def run_hvp(self, session, state, target, action, a_dist, vector, scene_scope):
        assert len(state) == len(target) == len(action) == len(a_dist)
        key = rl.get_key([self.network_scope, scene_scope])
        ret = session.run(self.hvp[key], feed_dict={
            self.s: state,
            self.t: target,
            self.a_old: action,
            self.a_dist_old: a_dist,
            self.v: vector,
            self.n: len(state),
        })
        return ret

    def get_vars(self, session, scene_scope):
        key = rl.get_key([self.network_scope, scene_scope])
        ret = session.run(self.get_op[key])
        return ret

    def set_vars(self, session, var_v, scene_scope):
        assert var_v.ndim == 1
        key = rl.get_key([self.network_scope, scene_scope])
        session.run(self.set_op[key], feed_dict={
            self.var_vs: var_v
        })
        return

    def run_ent(self, session, state, target, action, scene_scope):
        assert len(state) == len(target)
        key = rl.get_key([self.network_scope, scene_scope])
        ent = session.run(self.log_p[key], feed_dict={
            self.s: state,
            self.t: target,
            self.a_old: action,
            self.n: len(state),
        })
        return ent


class Network(object):
    """
        Modified network from ActorCriticFFNetwork for GAIL algorithm
    """
    def __init__(self,
                 config,
                 generator,
                 discriminator,
                 device="/cpu:0",
                 network_scope="network",
                 scene_scopes=("scene",)):
        self._device = device
        self.config = config
        self.g, self.d = generator, discriminator
        self.network_scope, self.scene_scopes = network_scope, scene_scopes
        self.losses, self.train_steps, self.evals, self.summaries, self.actions = {}, {}, {}, {}, {}
        tf.logging.debug('config=%s', self.config)

        # connect to discriminator
        discriminator.build_graph(self.g.s, self.g.t)


def test_discriminator(session, model, config, scene_scope, summary_writer):
    print("testing discriminator")
    batch_size = config.batch_size
    s_a = np.random.rand(batch_size*2, 2048, 4)
    t_a = np.random.rand(batch_size*2, 2048, 4)
    a_a = np.random.randint(0, high=config.action_size, size=(batch_size*2,))
    s_e = np.random.rand(batch_size/2, 2048, 4)
    t_e = np.random.rand(batch_size/2, 2048, 4)
    a_e = np.random.randint(0, high=config.action_size, size=(batch_size/2,))
    max_iter = 20
    for i in range(max_iter):
        loss = model.d.step_d(session, scene_scope, s_a, t_a, a_a, s_e, t_e, a_e, writer=summary_writer)
        rewards_a = float(np.mean(model.d.run_reward(session, scene_scope, s_a, t_a, a_a)))
        rewards_e = float(np.mean(model.d.run_reward(session, scene_scope, s_e, t_e, a_e)))
        print("%(i)d loss=%(loss)f rewards_e=%(rewards_e)f rewards_a=%(rewards_a)f" % locals())


def test_policy_trpo(session, model, config, scene_scope, summary_writer):
    print("testing policy network for trpo update")
    batch_size = config.batch_size
    n_data = 3 * batch_size + 1
    s_a = np.random.rand(n_data, 2048, 4) - 0.5
    t_a = np.random.rand(n_data, 2048, 4) - 0.5
    adv = np.random.rand(n_data)
    max_iter = 10
    var_vs = model.g.get_vars(session, scene_scope)
    print("get_vars return shape %s" % str(var_vs.shape))
    var_set_vs = np.random.rand(*var_vs.shape)
    model.g.set_vars(session, var_set_vs, scene_scope)
    var_vs = model.g.get_vars(session, scene_scope)
    assert np.allclose(var_vs, var_set_vs)
    a_old, a_dist_old = model.g.run_policy(session, s_a, t_a, scene_scope)
    a, a_dist = model.g.run_policy(session, s_a, t_a, scene_scope)
    assert np.allclose(a_dist, a_dist_old)
    sur_obj, kl = model.g.run_sur_obj_kl(session, s_a, t_a, a_old, a_dist_old, adv, scene_scope)
    print("sur_obj=%(sur_obj)f, kl=%(kl)f" % locals())
    max_iter = 10
    for i in range(max_iter):
        a_old = (a_old + 1) % config.action_size
        sur_obj, obj_grad, kl = model.g.run_sur_obj_kl_with_grads(session, s_a, t_a, a_old, a_dist_old, adv, scene_scope)
        print("sur_obj=%(sur_obj)f, kl=%(kl)f" % locals())
        obj_grad_norm = np.sum(obj_grad ** 2)
        print("obj_grad norm = %(obj_grad_norm).2f" % locals())
        # assert obj_grad_norm > 1e-1, "zero obj_grad_norm"
    vector = np.random.rand(*var_vs.shape)
    hvp = model.g.run_hvp(session, s_a, t_a, a, a_dist, vector, scene_scope)
    assert hvp.shape == vector.shape
    ent = model.g.run_ent(session, s_a, t_a, a, scene_scope)
    print(ent)
    assert all(ent >= 0)


def test_value_net(session, model, config, scene_scope, summary_writer):
    print("testing value network")
    batch_size = config.batch_size
    n_data = 3 * batch_size + 1
    s = np.random.rand(n_data, 2048, 4) - 0.5
    t = np.random.rand(n_data, 2048, 4) - 0.5
    target_value = np.random.rand(n_data)
    max_iter = 10
    for _ in range(max_iter):
        loss = model.g.step_value(session, s, t, target_value, summary_writer, scene_scope)
        print("%(loss)f" % locals())


def test_policy_supervised(session, model, config, scene_scope, summary_writer):
    print("testing policy network for supervised learning")
    batch_size = config.batch_size
    n_data = 3 * batch_size + 1
    s = np.random.rand(n_data, 2048, 4) - 0.5
    t = np.random.rand(n_data, 2048, 4) - 0.5
    a_dist_target = np.random.rand(n_data, 4)
    max_iter = 50
    for _ in range(max_iter):
        acc, loss = model.g.step_policy(session, s, t, a_dist_target, summary_writer, scene_scope)
        print("acc=%(acc)f loss=%(loss)f" % locals())
    acc, loss = model.g.eval_policy(session, s, t, a_dist_target, scene_scope)
    print("acc=%(acc)f loss=%(loss)f" % locals())


def test_model():
    config = Configuration()
    config.lr = 3e-5
    scene_scopes = ('scene1', 'scene2', 'scene3', 'scene4')
    train_logdir = 'logdir/ut_network'
    discriminator = Discriminator(config, network_scope='discriminator', scene_scopes=scene_scopes)
    generator = Generator(config, network_scope='generator', scene_scopes=scene_scopes)
    model = Network(config, generator, discriminator, network_scope='network', scene_scopes=scene_scopes)
    scene_scope = 'scene3'

    sess_config = tf.ConfigProto(log_device_placement=False,
                                 allow_soft_placement=True)
    for test_fn in (test_policy_trpo, test_discriminator, test_value_net, test_policy_supervised):
        with tf.Session(config=sess_config) as session:
            summary_writer = tf.summary.FileWriter(train_logdir, session.graph)
            session.run(tf.global_variables_initializer())
            test_fn(session, model, config, scene_scope, summary_writer)
        summary_writer.close()

if __name__ == '__main__':
    test_model()
