import tensorflow as tf
from tensorflow.contrib import layers
import logging
import math
import numpy as np
from config import Configuration

from utils import nn, rl

tf.logging.set_verbosity(tf.logging.DEBUG)


# Discriminator that distinguish between (s,a) of expert from generator
class Discriminator(object):
    def __init__(self,
                 config,
                 device="/cpu:0",
                 network_scope="network",
                 scene_scopes=("scene",)):
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
        self.config, self._device, self.network_scope, self.scene_scopes = \
            config, device, network_scope, scene_scopes
        self.lr = tf.placeholder(tf.float32, [], name='lr')
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.logits, self.rewards, self.losses = {}, {}, {}
        self.train_vars, self.train_steps, self.summaries = {}, {}, {}
        self.s_a, self.t_a, self.a_a, self.s_e, self.t_e, self.a_e = None, None, None, None, None, None

    def get_global_step(self):
        return self.global_step.eval()

    def create(self, s, t, a, is_expert=False):
        """ score is probability (s,a) come from expert, i.e. expert is labeled as zero
        Also log(p) is reward/cost function c(s,a)
        """
        logits, rewards, train_vars = {}, {}, {}
        with tf.device(self._device):
            fc1_out = nn.siamese(s, t, 512, name='fc1')
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
                    logits[key] = tf.layers.dense(x, self.config.action_size, name='logits', activation=None,
                                                  kernel_initializer=layers.variance_scaling_initializer())
                    tf.logging.debug("%s-logits: shape %s", key, logits[key].get_shape())
                    if not is_expert:
                        scores = tf.gather_nd(logits[key],
                                              tf.stack((tf.range(self.config.batch_size), a), axis=1),
                                              name='scores')
                        tf.logging.debug("%s-scores: shape %s", key, scores.get_shape())
                        # cost(s,a) = log(D)
                        rewards[key] = tf.log(tf.sigmoid(scores), name='rewards')
                        tf.logging.debug("%s-rewards: shape %s", key, rewards[key].get_shape())
                    scene_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scene_scope)
                    tf.logging.debug('scene %s variables %s ', scene_scope, scene_variables)
                    if key not in self.train_vars:
                        tf.logging.debug('set train_vars for %s', key)
                        self.train_vars[key] = shared_variables + scene_variables
        return logits, rewards

    def add_loss_op(self, s_a, t_a, a_a, s_e, t_e, a_e):
        self.s_a, self.t_a, self.a_a, self.s_e, self.t_e, self.a_e = s_a, t_a, a_a, s_e, t_e, a_e
        with tf.variable_scope("") as scope:
            logits_e, _ = self.create(s_e, t_e, a_e, is_expert=True)
            # Re-use discriminator weights on new inputs
            scope.reuse_variables()
            logits_a, self.rewards = self.create(s_a, t_a, a_a, is_expert=False)

        for scene_scope in self.scene_scopes:
            key = rl.get_key([self.network_scope, scene_scope])
            with tf.name_scope('loss/' + scene_scope):
                self.losses[key] = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(logits_e[key]), logits=logits_e[key]))
                self.losses[key] += tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(logits_a[key]), logits=logits_a[key]))
                self.summaries[key] = tf.summary.scalar("loss", self.losses[key])
            with tf.name_scope('optimizer/' + scene_scope):
                optimizer = tf.train.AdamOptimizer(learning_rate=self.lr, name='adam')
                self.train_steps[key] = optimizer.minimize(self.losses[key], global_step=self.global_step,
                                                           name='train',
                                                           var_list=self.train_vars[key])
        return

    def run_reward(self, session, scopes, s_a, t_a):
        assert len(s_a) == len(t_a)
        key = rl.get_key(scopes[:2])
        feed_dict = {
            self.s_a: s_a,
            self.t_a: t_a,
            self.lr: self.config.lr,
            self.is_training: False,
        }
        rewards = session.run([self.rewards[key]], feed_dict=feed_dict)[0]
        return rewards

    def step(self, session, scopes, s_a, t_a, s_e, t_e, a_e, writer=None):
        assert len(s_a) == len(t_a)
        assert len(s_e) == len(t_e) == len(a_e)
        key = rl.get_key(scopes[:2])
        n_data_a = len(s_a)
        n_data_e = len(s_e)
        batch_size = self.config.batch_size

        summary_op = self.summaries[key] if writer else tf.no_op()
        ops = [self.losses[key], self.train_steps[key], summary_op]

        total_loss = 0.0
        train_steps = int(math.ceil(n_data_a / batch_size))
        for i in range(train_steps):
            # generate indicies for the batch
            start_idx_a = (i * batch_size) % n_data_a
            end_idx_a = min(start_idx_a+batch_size, n_data_a)
            actual_batch_size = end_idx_a-start_idx_a
            start_idx_e = (i * batch_size) % n_data_e
            end_idx_e = min(start_idx_e+batch_size, n_data_e)
            feed_dict = {
                self.s_a: s_a[start_idx_a:end_idx_a],
                self.t_a: t_a[start_idx_a:end_idx_a],
                self.s_e: s_e[start_idx_e:end_idx_e],
                self.t_e: t_e[start_idx_e:end_idx_e],
                self.a_e: a_e[start_idx_e:end_idx_e],
                self.lr: self.config.lr,
                self.is_training: True,
            }
            # get batch size
            loss, _, summary = session.run(ops, feed_dict=feed_dict)

            # aggregate performance stats
            total_loss += loss * actual_batch_size
            if writer:
                logging.info("writing summary")
                writer.add_summary(summary, global_step=self.get_global_step())
        total_loss /= n_data_a
        return total_loss


def test_model():
    config = Configuration()
    network_scope = 'network_scope'
    scene_scopes = ('scene1', 'scene2', 'scene3', 'scene4')
    train_logdir = 'logdir'
    model = Discriminator(config, network_scope=network_scope, scene_scopes=scene_scopes)
    sess_config = tf.ConfigProto(log_device_placement=False,
                                 allow_soft_placement=True)
    batch_size = config.batch_size
    s_a_op = tf.placeholder(tf.float32, [None, 2048, 4], name='observation_a')  # state (input) 4 frames of Res50 code
    t_a_op = tf.placeholder(tf.float32, [None, 2048, 4], name='target_a')  # target (input) 4 frames of Res50 code
    a_a_op = tf.constant(np.random.randint(0, high=config.action_size, size=(batch_size,)), dtype=tf.int32)
    s_e_op = tf.placeholder(tf.float32, [None, 2048, 4], name='observation_e')  # state (input) 4 frames of Res50 code
    t_e_op = tf.placeholder(tf.float32, [None, 2048, 4], name='target_e')  # target (input) 4 frames of Res50 code
    a_e_op = tf.placeholder(tf.int32, [None], name='target_e')
    model.add_loss_op(s_a_op, t_a_op, a_a_op, s_e_op, t_e_op, a_e_op)

    s_a = np.random.rand(batch_size, 2048, 4)
    t_a = np.random.rand(batch_size, 2048, 4)
    s_e = np.random.rand(batch_size, 2048, 4)
    t_e = np.random.rand(batch_size, 2048, 4)
    a_e = np.random.randint(0, high=config.action_size, size=(batch_size,))
    max_iter = 100
    with tf.Session(config=sess_config) as session:
        summary_writer = tf.summary.FileWriter(train_logdir, session.graph)
        session.run(tf.global_variables_initializer())
        for i in range(max_iter):
            scope = (network_scope, 'scene2', 'task1')
            loss = model.step(session, scope, s_a, t_a, s_e, t_e, a_e, writer=summary_writer)
            rewards = float(np.mean(model.run_reward(session, scope, s_a, t_a)))
            print("%(i)d loss=%(loss)f rewards=%(rewards)f" % locals())
    summary_writer.close()

if __name__ == '__main__':
    test_model()
