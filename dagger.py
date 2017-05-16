import argparse
import math
import logging
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


def create_threads(config, model, network_scope, list_of_tasks):
    scene_scopes = list_of_tasks.keys()
    branches = [(scene, task) for scene in scene_scopes for task in list_of_tasks[scene]]
    return [DaggerThread(config, model, i, network_scope=network_scope, scene_scope=scene_scope, task_scope=task)
            for i, (scene_scope, task) in enumerate(branches)]


def anneal_lr(lr_config, global_t):
    """ heuristically update lr"""
    lr_schedules = ((2e3, 1.0), (1e4, 0.5), (5e4, 0.5**2), (1e5, 0.5**3), (5e5, 0.5**4), (1e6, 0.5**5))
    for step, rate in lr_schedules:
        if global_t < step:
            return lr_config * rate
    return lr_config * lr_schedules[-1][1]


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
        load_weights = True
    else:
        logging.info("initializing all variables")
        session.run(tf.global_variables_initializer())
        load_weights = False
    global_t = 0
    t0 = time.time()
    lr_config = config.lr
    iteration = 0
    while global_t < config.max_global_time_step:
        for thread in threads:
            if load_weights:
                thread.first_iteration = False
            global_t += thread.process(session, global_t, summary_writer)
            iteration += 1
            if saver and iteration % config.steps_per_save == (config.steps_per_save-1):
                logging.info('Save checkpoint at timestamp %d' % global_t)
                saver.save(session, weight_path)
                checkpoint = tf.train.get_checkpoint_state(weight_root)
                assert checkpoint and checkpoint.model_checkpoint_path
                saver.restore(session, checkpoint.model_checkpoint_path)
            if iteration % config.steps_per_eval == (config.steps_per_eval-1):
                evaluate_model(session, config, thread.local_network, summary_writer)
            thread.local_network.config.lr = anneal_lr(lr_config, global_t)
            logging.debug("lr=%f" % thread.local_network.config.lr)
    duration = time.time() - t0
    logging.info("global_t=%d and each step takes %0.2f s (%0.2f)" % (global_t, duration/global_t, duration))
    if saver:
        saver.save(session, weight_path)
    if summary_writer:
        summary_writer.close()


def train_models(configs):
    device = "/gpu:0" if USE_GPU else "/cpu:0"
    network_scope = TASK_TYPE
    list_of_tasks = TASK_LIST
    scene_scopes = list_of_tasks.keys()
    for config_dict in configs:
        config = Configuration(**config_dict)
        print("training with config=%s" % (str(config)))
        # build the model graph
        model = PolicyNetwork(config,
                              device=device,
                              network_scope=network_scope,
                              scene_scopes=scene_scopes)
        sess_config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
        threads = create_threads(config, model, network_scope, list_of_tasks)
        with tf.Session(config=sess_config) as session:
            train_model(session, config, threads, config_dict['logdir'], config_dict['weight_root'])
        tf.reset_default_graph()


def train():
    config_dict = vars(args)
    train_models([config_dict])


def search():
    config_dict = vars(args)
    for _ in range(args.max_attempt):
        # config_dict['lr'] = 10 ** np.random.uniform(-5, -3, size=1)[0]
        config_dict['lr'] = np.random.uniform(3e-4, 4e-4)
        train_models([config_dict])


def evaluate_model(session, config, model, summary_writer=None):
    network_scope = TASK_TYPE
    list_of_tasks = TASK_LIST
    threads = create_threads(config, model, network_scope, list_of_tasks)

    scene_stats = defaultdict(list)
    expert_stats = defaultdict(list)
    acc_stats = defaultdict(list)
    for thread in threads:
        scene = thread.scene_scope
        task = thread.task_scope
        lengths, collisions, accuracies = thread.evaluate(session, config.num_eval_episodes, expert_agent=False)
        exp_lengths, exp_collisions, _ = thread.evaluate(session, config.num_eval_episodes, expert_agent=True)
        logging.debug("Agent %s: mean_episode_length=%f/%f mean_episode_collision=%f/%f accuracies=%f" %
                      (scene+'/'+task, float(np.mean(lengths)),
                       float(np.mean(exp_lengths)),
                       float(np.mean(collisions)),
                       float(np.mean(exp_collisions)),
                       float(np.mean(accuracies)) ))
        scene_stats[scene] += lengths
        expert_stats[scene] += exp_lengths
        acc_stats[scene] += accuracies
        if summary_writer:
            summary_values = {
                "stats/" + scene + "-steps": float(np.mean(scene_stats[scene])),
                "stats/" + scene + "-accuracy": float(np.mean(acc_stats[scene])),
            }
            thread.add_summary(summary_writer, summary_values)
    logging.info("Average_trajectory_length per scene (steps):")
    for scene in scene_stats:
        logging.info("%s: agent=%f (acc=%.2f) expert=%f" %
                     (scene,
                      float(np.mean(scene_stats[scene])),
                      float(np.mean(acc_stats[scene])),
                      float(np.mean(expert_stats[scene])),
                      ))


def evaluate():
    device = "/gpu:0" if USE_GPU else "/cpu:0"
    network_scope = TASK_TYPE
    list_of_tasks = TASK_LIST
    scene_scopes = list_of_tasks.keys()
    weight_root = args.weight_root
    config = Configuration(**vars(args))
    print("evaluating with config=%s" % (str(config)))
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
        evaluate_model(session, config, model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-l', dest='log_level', default='info', help="logging level: {debug, info, error}")
    parser.add_argument('--max_attempt', dest='max_attempt', type=int, default=1,
                        help='search hyper parameters')
    parser.add_argument('--logdir', dest='logdir', default='./logdir', help='logdir')
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

    # search
    parser_train = subparser.add_parser('search', help='search hyper parameters')
    parser_train.set_defaults(func=search)

    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format='%(asctime)s %(levelname)s %(message)s')
    if args.command:
        args.func()
    else:
        parser.print_help()


