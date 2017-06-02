# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.contrib import layers
import logging
import math
import numpy as np
from utils import nn, rl
from config import Configuration
from gail.discriminator import Discriminator

tf.logging.set_verbosity(tf.logging.DEBUG)


# generator
class Generator(object):
    def __init__(self,
                 config,
                 device='/cpu:0',
                 name='generator',
                 network_scope='network',
                 scene_scopes=('scene',)):
        self._device = device
        self.config = config
        self.network_scope, self.scene_scopes = network_scope, scene_scopes
        tf.logging.debug('config=%s', self.config)

        self.evals, self.summaries, self.actions = {}, {}, {}
        self.train_vars, self.actions_dists, self.values = {}, {}, {}
        with tf.variable_scope(name):
            self.s = tf.placeholder(tf.float32, [None, 2048, 4], name='observation')
            self.t = tf.placeholder(tf.float32, [None, 2048, 4], name='target')
            self.is_training = tf.placeholder(tf.bool, name='is_training')
            self.lr = tf.placeholder(tf.float32, [], name='lr')
            self.build_graph()

    def build_graph(self):
        with tf.device(self._device):
            fc1_out = nn.siamese(self.s, self.t, 512, name='fc1')
            tf.logging.debug("fc1: shape %s", fc1_out.get_shape())
            # shared fusion layer
            fc2_out = tf.layers.dense(fc1_out, 512, name='fc2', activation=nn.leaky_relu,
                                      kernel_initializer=layers.variance_scaling_initializer())
            tf.logging.debug("fc2: shape %s", fc2_out.get_shape())
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
                    # policy output layer
                    logits = tf.layers.dense(x, self.config.action_size, name='action_logits', activation=None,
                                             kernel_initializer=layers.variance_scaling_initializer())
                    self.actions_dists[key] = tf.nn.softmax(logits=logits)
                    tf.logging.debug("%s-action_dist: shape %s", key, self.actions_dists[key].get_shape())
                    self.actions[key] = tf.cast(tf.reshape(
                        tf.argmax(logits, axis=1), (-1,)), tf.int32, name='action_a')
                    tf.logging.debug("%s-action: shape %s", key, self.actions[key].get_shape())
                    # value output layer
                    values = tf.layers.dense(x, 1, name='value_logits', activation=None,
                                             kernel_initializer=layers.variance_scaling_initializer())
                    self.values[key] = tf.reshape(values, (-1,))
                    tf.logging.debug("%s-values: shape %s", key, self.values[key].get_shape())
                    scene_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scene_scope)
                    tf.logging.debug('scene %s variables %s ', scene_scope, scene_variables)
                    if key not in self.train_vars:
                        self.train_vars[key] = scene_variables

    def run_policy(self, session, state, target, key):
        a, a_dist = session.run([self.actions[key], self.actions_dists[key]], feed_dict={
            self.s: state,
            self.t: target,
            self.lr: self.config.lr,
            self.is_training: False,
        })
        return a, a_dist


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
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
        tf.logging.debug('config=%s', self.config)

        # connect to discriminator
        discriminator.build_graph(self.g.s, self.g.t)

    def get_global_step(self):
        return self.global_step.eval()

    def step_d(self, session, key, s_a, t_a, a_a, s_e, t_e, a_e, writer=None):
        assert len(s_a) == len(t_a) == len(a_a)
        assert len(s_e) == len(t_e) == len(a_e)
        n_data_a = len(s_a)
        n_data_e = len(s_e)
        batch_size = self.config.batch_size

        summary_op = self.d.summaries[key] if writer else tf.no_op()
        ops = [self.d.losses[key], self.d.train_steps[key], summary_op]

        total_loss = 0.0
        train_steps = int(math.ceil(n_data_a / batch_size))
        for i in range(train_steps):
            # generate indicies for the batch
            start_idx_a = (i * batch_size) % n_data_a
            end_idx_a = min(start_idx_a+batch_size, n_data_a)
            actual_size = end_idx_a-start_idx_a
            start_idx_e = (i * batch_size) % n_data_e
            end_idx_e = min(start_idx_e+batch_size, n_data_e)
            feed_dict = {
                self.g.s: s_a[start_idx_a:end_idx_a],
                self.g.t: t_a[start_idx_a:end_idx_a],
                self.d.s_e: s_e[start_idx_e:end_idx_e],
                self.d.t_e: t_e[start_idx_e:end_idx_e],
                self.d.a_e[key]: a_e[start_idx_e:end_idx_e],
                self.d.a_a[key]: a_a[start_idx_e:end_idx_e],
                self.g.lr: self.config.lr,
                self.g.is_training: True,
                self.d.lr: self.config.lr,
                self.d.is_training: True,
                self.d.n: actual_size,
            }
            # get batch size
            loss, _, summary = session.run(ops, feed_dict=feed_dict)

            # aggregate performance stats
            total_loss += loss * actual_size
            if writer:
                logging.info("writing summary")
                writer.add_summary(summary, global_step=self.get_global_step())
        total_loss /= n_data_a
        return total_loss

    def run_reward(self, session, key, s_a, t_a, a_a):
        assert len(s_a) == len(t_a) == len(a_a)
        batch_size = self.config.batch_size
        n_data = len(s_a)
        n_steps = int(math.ceil(n_data / batch_size))
        rewards = []
        for i in range(n_steps):
            # generate indicies for the batch
            start_idx = i * batch_size
            end_idx = min(start_idx+batch_size, n_data)
            actual_size = end_idx - start_idx
            feed_dict = {
                self.g.s: s_a[start_idx:end_idx],
                self.g.t: t_a[start_idx:end_idx],
                self.d.a_a[key]: a_a[start_idx:end_idx],
                self.d.n: actual_size,
                self.g.lr: self.config.lr,
                self.g.is_training: False,
            }
            r = session.run([self.d.rewards[key]], feed_dict=feed_dict)[0]
            rewards.append(r)
        rewards = np.concatenate(rewards, axis=0)
        return rewards


def test_model():
    config = Configuration()
    network_scope = 'network_scope'
    scene_scopes = ('scene1', 'scene2', 'scene3', 'scene4')
    train_logdir = 'logdir'
    discriminator = Discriminator(config, network_scope=network_scope, scene_scopes=scene_scopes)
    generator = Generator(config, network_scope=network_scope, scene_scopes=scene_scopes)
    model = Network(config, generator, discriminator, network_scope=network_scope, scene_scopes=scene_scopes)

    batch_size = config.batch_size
    a_e, a_a = {}, {}
    for scene_scope in scene_scopes:
        key = rl.get_key([network_scope, scene_scope])
        a_e[key] = np.random.randint(0, high=config.action_size, size=(batch_size/2,))
        a_a[key] = np.random.randint(0, high=config.action_size, size=(batch_size/2,))
    s_a = np.random.rand(batch_size/2, 2048, 4)
    t_a = np.random.rand(batch_size/2, 2048, 4)
    s_e = np.random.rand(batch_size/2, 2048, 4)
    t_e = np.random.rand(batch_size/2, 2048, 4)
    max_iter = 100
    key = rl.get_key([network_scope, 'scene2'])
    sess_config = tf.ConfigProto(log_device_placement=False,
                                 allow_soft_placement=True)
    with tf.Session(config=sess_config) as session:
        summary_writer = tf.summary.FileWriter(train_logdir, session.graph)
        session.run(tf.global_variables_initializer())
        for i in range(max_iter):
            loss = model.step_d(session, key, s_a, t_a, a_a[key], s_e, t_e, a_e[key], writer=summary_writer)
            rewards = float(np.mean(model.run_reward(session, key, s_a, t_a, a_a[key])))
            print("%(i)d loss=%(loss)f rewards=%(rewards)f" % locals())
        a, a_dist = model.g.run_policy(session, s_a, t_a, key)
        print(a_dist[0])
    summary_writer.close()

if __name__ == '__main__':
    test_model()
