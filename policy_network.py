# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.contrib import layers
import math
import numpy as np

tf.logging.set_verbosity(tf.logging.INFO)


# Policy only network
class PolicyNetwork(object):
    """
        Modified network from ActorCriticFFNetwork for DAgger algorithm
    """
    def __init__(self,
                 config,
                 device="/cpu:0",
                 network_scope="network",
                 scene_scopes=("scene")):
        self._device = device
        self.config = config
        self.losses = dict()
        self.scores = dict()
        self.train_steps = dict()
        self.evals = dict()
        self.summaries = dict()
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
        tf.logging.debug('config=%s', self.config)

        with tf.device(self._device):
            self.s = tf.placeholder(tf.float32, [None, 2048, 4], name='observation')  # state (input) 4 frames of Res50 code
            self.t = tf.placeholder(tf.float32, [None, 2048, 4], name='target')  # target (input) 4 frames of Res50 code
            self.y = tf.placeholder(tf.float32, [None, self.config.action_size], name='action')  # label
            self.is_training = tf.placeholder(tf.bool, name='is_training')
            self.lr = tf.placeholder(tf.float32, [], name='lr')

            fc1_out = self._siamese(self.s, self.t, 512, name='fc1')
            tf.logging.debug("fc1: shape %s", fc1_out.get_shape())
            fc2_out = self._fc(fc1_out, 512, name='fc2')   # shared fusion layer
            tf.logging.debug("fc2: shape %s", fc2_out.get_shape())
            shared_variables = tf.trainable_variables()
            tf.logging.debug('shared variables %s ', shared_variables)
            for scene_scope in scene_scopes:
                # scene-specific key
                key = self._get_key([network_scope, scene_scope])
                with tf.variable_scope(scene_scope) as scope:
                    # scene-specific adaptation layer, disable bn to make it easier for optimizer op dependency
                    x = self._fc(fc2_out, 512, name='fc3')
                    tf.logging.debug("%s-fc3: shape %s", key, x.get_shape())
                    # policy output layer
                    logits = self._fc(x, self.config.action_size, name='logits', activation=False)
                    self.scores[key] = tf.nn.softmax(logits=logits)
                    tf.logging.debug("%s-out: shape %s", key, self.scores[key].get_shape())
                    scene_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scene_scope)
                    tf.logging.debug('scene %s variables %s ', scene_scope, scene_variables)
                with tf.name_scope('loss/' + scene_scope):
                    ce = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.y)
                    data_loss = tf.reduce_mean(ce, name='loss')
                    self.losses[key] = data_loss  #  + _add_reg(self.config.reg)
                    self.summaries[key] = tf.summary.scalar("loss", self.losses[key])
                    # batch normalization in tensorflow requires this extra dependency
                    # extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                    # with tf.control_dependencies(extra_update_ops):
                with tf.name_scope('optimizer/' + scene_scope):
                    optimizer = tf.train.AdamOptimizer(learning_rate=self.lr, name='adam')
                    self.train_steps[key] = optimizer.minimize(self.losses[key], global_step=self.global_step,
                                                               name='train',
                                                               var_list=shared_variables + scene_variables)
                with tf.name_scope('eval/' + scene_scope):
                    labels = tf.argmax(self.scores[key], 1)
                    ground_truth_labels = tf.argmax(self.y, 1)
                    self.evals[key] = tf.cast(tf.equal(labels, ground_truth_labels), tf.float32)

    def run_policy(self, session, state, target, scopes):
        k = self._get_key(scopes[:2])
        pi_out = session.run(self.scores[k], feed_dict={
            self.s: [state],
            self.t: [target],
            self.lr: self.config.lr,
            self.is_training: False,
        })
        return pi_out[0]

    def run_epoch(self, session, scopes, s, t, y, training_now, writer=None):
        key = self._get_key(scopes[:2])
        assert len(s) == len(t) == len(y)
        n_data = len(s)
        batch_size = self.config.batch_size

        summary_op = self.summaries[key] if writer else tf.no_op()
        extra_op = self.train_steps[key] if training_now else tf.no_op()
        ops = [self.losses[key], self.evals[key], summary_op, extra_op]

        total_loss = 0.0
        total_correct = 0
        train_steps = int(math.ceil(n_data / batch_size))
        for i in range(train_steps):
            # generate indicies for the batch
            start_idx = (i * batch_size) % n_data
            end_idx = min(start_idx+batch_size, n_data)
            actual_batch_size = end_idx-start_idx
            # create a feed dictionary for this batch
            feed_dict = {self.s: s[start_idx:end_idx],
                         self.t: t[start_idx:end_idx],
                         self.y: y[start_idx:end_idx],
                         self.lr: self.config.lr,
                         self.is_training: training_now}
            # get batch size
            loss, corr, summary, _ = session.run(ops, feed_dict=feed_dict)

            # aggregate performance stats
            total_loss += loss * actual_batch_size
            total_correct += np.sum(corr)
            if writer:
                writer.add_summary(summary, global_step=self.get_global_step())
        total_correct /= n_data
        total_loss /= n_data
        return total_loss, total_correct

    def _get_key(self, scopes):
        return '/'.join(scopes)

    def _fc_weight_variable(self, shape, name='W_fc'):
        input_channels = shape[0]
        d = 1.0 / np.sqrt(input_channels)
        initial = tf.random_uniform(shape, minval=-d, maxval=d)
        return tf.Variable(initial, name=name)

    def _fc_bias_variable(self, shape, input_channels, name='b_fc'):
        d = 1.0 / np.sqrt(input_channels)
        initial = tf.random_uniform(shape, minval=-d, maxval=d)
        return tf.Variable(initial, name=name)

    def _relu(self, x):
        return tf.maximum(self.config.relu_leakiness*x, x)

    def _siamese(self, x1, x2, out_dim, name):
        # flatten input
        x1_size = np.prod(x1.get_shape().as_list()[1:])
        x2_size = np.prod(x2.get_shape().as_list()[1:])
        assert x1_size == x2_size
        with tf.variable_scope(name):
            x1 = tf.reshape(x1, [-1, x1_size], name='x1')
            x2 = tf.reshape(x2, [-1, x1_size], name='x2')
            w = tf.get_variable('weight', (x1_size, out_dim), initializer=layers.variance_scaling_initializer())
            b = tf.get_variable('bias', (out_dim,), initializer=tf.constant_initializer())
            x1 = tf.matmul(x1, w) + b
            # x1 = layers.batch_norm(x1, center=True, scale=True, is_training=self.is_training)
            x1 = self._relu(x1)
            x2 = tf.matmul(x2, w) + b
            # x2 = layers.batch_norm(x2, center=True, scale=True, is_training=self.is_training)
            x2 = self._relu(x2)
            x = tf.concat(values=[x1, x2], axis=1)
        return x

    def _fc(self, x, out_dim, name, activation=True):
        with tf.variable_scope(name):
            w = tf.get_variable('weight', (x.get_shape()[1], out_dim), initializer=layers.variance_scaling_initializer())
            b = tf.get_variable('bias', (out_dim,), initializer=tf.constant_initializer())
            x = tf.matmul(x, w) + b
            if activation:
                # x = layers.batch_norm(x, center=True, scale=True, is_training=self.is_training) if use_bn else x
                x = self._relu(x)
        return x

    def get_global_step(self):
        return self.global_step.eval()
