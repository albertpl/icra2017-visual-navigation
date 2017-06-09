import tensorflow as tf
from tensorflow.contrib import layers
import logging
import math
from utils import nn, rl

tf.logging.set_verbosity(tf.logging.INFO)


# Discriminator that distinguish between (s,a) of expert from generator
class Discriminator(object):
    def __init__(self,
                 config,
                 device='/cpu:0',
                 network_scope='discriminator',
                 scene_scopes=('scene',)):
        self.config, self._device, self.network_scope, self.scene_scopes = \
            config, device, network_scope, scene_scopes
        self.global_step, self.lr, self.is_training = None, None, None
        self.logits, self.rewards, self.losses = {}, {}, {}
        self.train_vars, self.train_steps, self.summaries = {}, {}, {}
        self.s_e, self.t_e, self.a_e, self.n_e = None, None, None, None
        self.s_a, self.t_a, self.a_a, self.n_a = None, None, None, None

    def get_global_step(self):
        return self.global_step.eval()

    def create(self, s, t, a, n, scope):
        """ score is probability (s,a) come from expert, i.e. expert is labeled as zero
        Also log(p) is reward/cost function c(s,a)
        """
        logits, rewards, train_vars = {}, {}, {}
        with tf.device(self._device):
            fc1_out = nn.siamese(s, t, 128, name='fc1')
            tf.logging.debug("%s fc1: shape %s", self.network_scope, fc1_out.get_shape())
            # shared fusion layer
            fc2_out = tf.layers.dense(fc1_out, 128, name='fc2', activation=nn.leaky_relu,
                                      kernel_initializer=layers.variance_scaling_initializer())
            tf.logging.debug("%s fc2: shape %s", self.network_scope, fc2_out.get_shape())
            if len(self.train_vars) == 0:
                shared_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.name)
                tf.logging.debug('shared variables %s ', shared_variables)
            else:
                shared_variables = None
            for scene_scope in self.scene_scopes:
                # scene-specific key
                key = rl.get_key([self.network_scope, scene_scope])
                with tf.variable_scope(scene_scope):
                    # scene-specific adaptation layer, disable bn to make it easier for optimizer op dependency
                    x = tf.layers.dense(fc2_out, 128, name='fc3', activation=nn.leaky_relu,
                                        kernel_initializer=layers.variance_scaling_initializer())
                    tf.logging.debug("%s-fc3: shape %s", key, x.get_shape())
                    # policy output layer
                    out = tf.layers.dense(x, self.config.action_size, name='out', activation=None,
                                          kernel_initializer=layers.variance_scaling_initializer())
                    tf.logging.debug("%s-out: shape %s", key, out.get_shape())
                    logits[key] = tf.gather_nd(
                        out, tf.stack((tf.range(n), a), axis=1), name='logits')
                    tf.logging.debug("%s-logits: shape %s", key, logits[key].get_shape())
                    # cost(s,a) = log(D)
                    rewards[key] = tf.log(tf.sigmoid(logits[key]), name='rewards')
                    tf.logging.debug("%s-rewards: shape %s", key, rewards[key].get_shape())
                    if key not in self.train_vars:
                        assert shared_variables is not None
                        scene_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=key)
                        tf.logging.debug('set train_vars for %s', key)
                        self.train_vars[key] = shared_variables + scene_variables
                        tf.logging.debug('scene %s variables %s ', scene_scope, self.train_vars[key])
        return logits, rewards

    def build_graph(self, s_a, t_a):
        self.s_a, self.t_a = s_a, t_a
        with tf.variable_scope(self.network_scope) as scope:
            self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
            self.lr = tf.placeholder(tf.float32, [], name='lr')
            self.is_training = tf.placeholder(tf.bool, name='is_training')
            self.n_e = tf.placeholder(tf.int32, [], name='n_e')
            self.n_a = tf.placeholder(tf.int32, [], name='n_a')
            self.s_e = tf.placeholder(tf.float32, [None, 2048, 4], name='observation')
            self.t_e = tf.placeholder(tf.float32, [None, 2048, 4], name='target')
            self.a_e = tf.placeholder(tf.int32, [None], name='action')
            self.a_a = tf.placeholder(tf.int32, [None], name='action')
            logits_e, _ = self.create(self.s_e, self.t_e, self.a_e, self.n_e, scope)
            # Re-use discriminator weights on new inputs
            scope.reuse_variables()
            logits_a, self.rewards = self.create(self.s_a, self.t_a, self.a_a, self.n_a, scope)
            self.logits = logits_a
            # WGAN-GP loss
            eps = tf.random_uniform([], name='eps')
            n_hat = tf.minimum(self.n_a, self.n_e, name='n_hat')
            s_hat = eps * self.s_e[:n_hat] + (1-eps) * self.s_a[:n_hat]
            a_e_f = tf.cast(self.a_e[:n_hat], tf.float32, name='a_e_f')
            a_a_f = tf.cast(self.a_a[:n_hat], tf.float32, name='a_a_f')
            a_hat_f = eps * a_e_f + (1-eps) * a_a_f
            tf.logging.debug("a_hat_f %s ", a_hat_f)
            a_hat = tf.cast(tf.rint(a_hat_f), tf.int32, name='a_hat')
            tf.logging.debug("a_hat_f %s ", a_hat)
            logits_hat, _ = self.create(s_hat, self.t_a[:n_hat], a_hat, n_hat, scope)
            tf.logging.debug("logits_hat %s ", logits_hat)

        for scene_scope in self.scene_scopes:
            key = rl.get_key([self.network_scope, scene_scope])
            with tf.name_scope('loss_d/' + scene_scope):
                # self.losses[key] = tf.reduce_mean(
                #    tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(logits_e[key]) * 0.9,
                #                                            logits=logits_e[key]))
                #self.losses[key] += tf.reduce_mean(
                #    tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(logits_a[key]), logits=logits_a[key]))
                self.losses[key] = tf.reduce_mean(logits_a[key]) - tf.reduce_mean(logits_e[key])

                grad_hat = nn.flat_grad(logits_hat[key], self.train_vars[key])
                tf.logging.debug("%s-grad_hat: %s", key, grad_hat)
                grad_norm = tf.norm(grad_hat)
                grad_pen = tf.reduce_mean((grad_norm-1)**2)
                self.losses[key] += self.config.wgan_lam * grad_pen
                self.summaries[key] = tf.summary.scalar("loss", self.losses[key])
            with tf.name_scope('optimizer_d/' + scene_scope):
                optimizer = tf.train.AdamOptimizer(learning_rate=self.lr, name='adam')
                self.train_steps[key] = optimizer.minimize(self.losses[key], global_step=self.global_step,
                                                           name='train',
                                                           var_list=self.train_vars[key])
        return

    def step_d(self, session, scene_scope, s_a, t_a, a_a, s_e, t_e, a_e, writer=None):
        assert len(s_a) == len(t_a) == len(a_a)
        assert len(s_e) == len(t_e) == len(a_e)
        n_data_a = len(s_a)
        n_data_e = len(s_e)
        key = rl.get_key([self.network_scope, scene_scope])
        summary_op = self.summaries[key] if writer else tf.no_op()
        ops = [self.losses[key], self.train_steps[key], summary_op]
        # generate indicies for the batch
        feed_dict = {
            self.s_a: s_a,
            self.t_a: t_a,
            self.s_e: s_e,
            self.t_e: t_e,
            self.a_e: a_e,
            self.a_a: a_a,
            self.lr: self.config.lr,
            self.n_a: n_data_a,
            self.n_e: n_data_e,
        }
        # get batch size
        loss, _, summary = session.run(ops, feed_dict=feed_dict)
        if writer:
            writer.add_summary(summary, global_step=self.get_global_step())
        return loss

    def run_reward(self, session, scene_scope, s_a, t_a, a_a):
        assert len(s_a) == len(t_a) == len(a_a)
        key = rl.get_key([self.network_scope, scene_scope])
        feed_dict = {
            self.s_a: s_a,
            self.t_a: t_a,
            self.a_a: a_a,
            self.n_a: len(s_a),
            self.lr: self.config.lr,
        }
        return session.run(self.rewards[key], feed_dict=feed_dict)
