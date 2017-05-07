import argparse
import logging
import math
import numpy as np
import os
import tensorflow as tf
import time
from collections import defaultdict, Callable

from constants import ACTION_SIZE
from constants import PARALLEL_SIZE
from constants import INITIAL_ALPHA_LOW
from constants import INITIAL_ALPHA_HIGH
from constants import INITIAL_ALPHA_LOG_RATE
from constants import CHECKPOINT_DIR
from constants import LOG_FILE
from constants import RMSP_EPSILON
from constants import RMSP_ALPHA
from constants import GRAD_NORM_CLIP
from constants import USE_GPU
from constants import TASK_TYPE
from constants import TASK_LIST
from constants import NUM_EVAL_EPISODES

from config import Configuration
from policy_network import PolicyNetwork
from train_dagger import DaggerThread


def train_model(session, config, threads, logdir, weight_root):
    if logdir:
        if not os.path.exists(logdir):
            os.makedirs(logdir)
        test_n = len(list(n for n in os.listdir(logdir) if n.startswith('t')))
        train_logdir = logdir + '/t' + str(test_n + 1) + '_lr-' + str(config.lr)
        summary_writer = tf.summary.FileWriter(train_logdir, session.graph)
        logging.info("writing logs to %(train_logdir)s" % locals())
    else:
        summary_writer = None

    if weight_root:
        if not os.path.exists(weight_root):
            os.makedirs(weight_root)
        weight_path = weight_root + '/modelv1'
        logging.info("writing weights to %(weight_path)s" % locals())
        saver = tf.train.Saver()
        checkpoint = tf.train.get_checkpoint_state(weight_root)
    else:
        saver = weight_path = checkpoint = None
    if saver and checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(session, checkpoint.model_checkpoint_path)
    else:
        logging.info("initializing all variables")
        session.run(tf.global_variables_initializer())
    global_t = 0
    steps_per_save = 1e+4
    while global_t < config.max_global_time_step:
        for thread in threads:
            global_t += thread.process(session, global_t, summary_writer)
            if saver and global_t % steps_per_save == (steps_per_save-1):
                logging.info('Save checkpoint at timestamp %d' % global_t)
                saver.save(session, weight_path)
    if saver:
        saver.save(session, weight_path)
    if summary_writer:
        summary_writer.close()


def train_models(configs):
    device = "/gpu:0" if USE_GPU else "/cpu:0"
    network_scope = TASK_TYPE
    list_of_tasks = TASK_LIST
    scene_scopes = list_of_tasks.keys()
    branches = [(scene, task) for scene in scene_scopes for task in list_of_tasks[scene]]
    for config_dict in configs:
        config = Configuration(**config_dict)
        print("training with config=%s" % (str(config)))
        # build the model graph
        tf.reset_default_graph()
        model = PolicyNetwork(config,
                              device=device,
                              network_scope=network_scope,
                              scene_scopes=scene_scopes)
        sess_config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
        threads = [DaggerThread(config, model, i,
                                network_scope=network_scope,
                                scene_scope=scene_scope,
                                task_scope=task)
                   for i, (scene_scope, task) in enumerate(branches)]
        with tf.Session(config=sess_config) as session:
            train_model(session, config, threads, config_dict['logdir'], config_dict['weight_root'])


def train():
    config_dict = vars(args)
    for _ in range(args.max_attempt):
        config_dict['lr'] = 10 ** np.random.uniform(-5, -3, size=1)[0]
        train_models([config_dict])


def evaluate():
    device = "/gpu:0" if USE_GPU else "/cpu:0"
    weight_root = args.weight_root
    network_scope = TASK_TYPE
    list_of_tasks = TASK_LIST
    scene_scopes = list_of_tasks.keys()
    scene_stats = defaultdict(list)
    config = Configuration(**vars(args))
    model = PolicyNetwork(config,
                          device=device,
                          network_scope=network_scope,
                          scene_scopes=scene_scopes)
    sess_config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    with tf.Session(config=sess_config) as session:
        weight_path = weight_root + '/modelv1'
        saver = tf.train.Saver()
        checkpoint = tf.train.get_checkpoint_state(weight_root)
        assert saver and checkpoint and checkpoint.model_checkpoint_path, weight_path + " doesn't exist"
        saver.restore(session, checkpoint.model_checkpoint_path)
        branches = [(scene, task) for scene in scene_scopes for task in list_of_tasks[scene]]
        for i, (scene, task) in enumerate(branches):
            thread = DaggerThread(config, model, i,
                                  network_scope=network_scope,
                                  scene_scope=scene,
                                  task_scope=task)
            lengths, collisions = thread.evaluate(session, NUM_EVAL_EPISODES)
            logging.info("evaluating %s: mean_episode_length=%f mean_episode_collision=%f"
                         % (scene+'/'+task, float(np.mean(lengths)), float(np.mean(collisions))))
            scene_stats[scene] += lengths
        logging.info("Average_trajectory_length per scene:")
        for scene in scene_stats:
            logging.info("%s: %f steps" % (scene, float(np.mean(scene_stats[scene]))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-l', dest='log_level', default='info', help="logging level: {debug, info, error}")
    parser.add_argument('--max_attempt', dest='max_attempt', type=int, default=1,
                        help='search hyper parameters')
    parser.add_argument('--logdir', dest='logdir', default='/tmp/logdir', help='logdir')
    parser.add_argument('--weight_root', dest='weight_root', default=None, help='weights')
    for k in [a for a in dir(Configuration()) if not isinstance(a, Callable) and not a.startswith("__")]:
        parser.add_argument('--'+k, dest=k)
    subparser = parser.add_subparsers(dest='command', title='sub_commands', description='valid commands')
    # train
    parser_train = subparser.add_parser('train', help='train')
    parser_train.set_defaults(func=train)

    # eval
    parser_eval = subparser.add_parser('eval', help='evaluate')
    parser_eval.set_defaults(func=evaluate)

    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format='%(asctime)s %(levelname)s %(message)s')
    if args.command:
        args.func()
    else:
        parser.print_help()


