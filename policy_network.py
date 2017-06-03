# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.contrib import layers
import math
import numpy as np
from utils import nn, rl

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
                 scene_scopes=("scene",)):
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

            fc1_out = nn.siamese(self.s, self.t, 512, name='fc1')
            tf.logging.debug("fc1: shape %s", fc1_out.get_shape())
            # shared fusion layer
            fc2_out = tf.layers.dense(fc1_out, 512, name='fc2', activation=nn.leaky_relu,
                                      kernel_initializer=layers.variance_scaling_initializer())
            tf.logging.debug("fc2: shape %s", fc2_out.get_shape())
            shared_variables = tf.trainable_variables()
            tf.logging.debug('shared variables %s ', shared_variables)
            for scene_scope in scene_scopes:
                # scene-specific key
                key = rl.get_key([network_scope, scene_scope])
                with tf.variable_scope(scene_scope) as scope:
                    # scene-specific adaptation layer, disable bn to make it easier for optimizer op dependency
                    x = tf.layers.dense(fc2_out, 512, name='fc3', activation=nn.leaky_relu,
                                        kernel_initializer=layers.variance_scaling_initializer())
                    tf.logging.debug("%s-fc3: shape %s", key, x.get_shape())
                    # policy output layer
                    logits = tf.layers.dense(x, self.config.action_size, name='logits', activation=None,
                                             kernel_initializer=layers.variance_scaling_initializer())
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
        k = rl.get_key(scopes[:2])
        pi_out = session.run(self.scores[k], feed_dict={
            self.s: [state],
            self.t: [target],
            self.lr: self.config.lr,
            self.is_training: False,
        })
        return pi_out[0]

    def run_epoch(self, session, scopes, s, t, y, training_now, writer=None):
        key = rl.get_key(scopes[:2])
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

    def get_global_step(self):
        return self.global_step.eval()
