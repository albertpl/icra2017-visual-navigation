import tensorflow as tf
from tensorflow.contrib import layers

from utils import nn, rl

tf.logging.set_verbosity(tf.logging.DEBUG)


# Discriminator that distinguish between (s,a) of expert from generator
class Discriminator(object):
    def __init__(self,
                 config,
                 device='/cpu:0',
                 name='discriminator',
                 network_scope='network',
                 scene_scopes=('scene',)):
        self.config, self._device, self.network_scope, self.scene_scopes, self.name = \
            config, device, network_scope, scene_scopes, name
        self.global_step, self.lr, self.is_training = None, None, None
        self.logits, self.rewards, self.losses = {}, {}, {}
        self.train_vars, self.train_steps, self.summaries = {}, {}, {}
        self.s_e, self.t_e = None, None
        self.s_a, self.t_a, self.a_a = None, None, None
        self.a_e = {}

    def get_global_step(self):
        return self.global_step.eval()

    def create(self, s, t, actions, is_expert=False):
        """ score is probability (s,a) come from expert, i.e. expert is labeled as zero
        Also log(p) is reward/cost function c(s,a)
        """
        assert isinstance(actions, dict)
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
                assert key in actions
                a = actions[key]
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

    def build_graph(self, s_a, t_a, a_a):
        self.s_a, self.t_a, self.a_a = s_a, t_a, a_a
        with tf.variable_scope(self.name) as scope:
            self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
            self.lr = tf.placeholder(tf.float32, [], name='lr')
            self.is_training = tf.placeholder(tf.bool, name='is_training')
            self.s_e = tf.placeholder(tf.float32, [None, 2048, 4], name='observation')
            self.t_e = tf.placeholder(tf.float32, [None, 2048, 4], name='target')
            for scene_scope in self.scene_scopes:
                key = rl.get_key([self.network_scope, scene_scope])
                self.a_e[key] = tf.placeholder(tf.int32, [None], name='action')  # label
            logits_e, _ = self.create(self.s_e, self.t_e, self.a_e, is_expert=True)
            # Re-use discriminator weights on new inputs
            scope.reuse_variables()
            logits_a, self.rewards = self.create(s_a, t_a, a_a, is_expert=False)

        for scene_scope in self.scene_scopes:
            key = rl.get_key([self.network_scope, scene_scope])
            with tf.name_scope('loss_d/' + scene_scope):
                self.losses[key] = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(logits_e[key]), logits=logits_e[key]))
                self.losses[key] += tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(logits_a[key]), logits=logits_a[key]))
                self.summaries[key] = tf.summary.scalar("loss", self.losses[key])
            with tf.name_scope('optimizer_d/' + scene_scope):
                optimizer = tf.train.AdamOptimizer(learning_rate=self.lr, name='adam')
                self.train_steps[key] = optimizer.minimize(self.losses[key], global_step=self.global_step,
                                                           name='train',
                                                           var_list=self.train_vars[key])
        return

