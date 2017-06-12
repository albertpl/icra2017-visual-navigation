import argparse
import logging
import numpy as np
import os
import tensorflow as tf
import time
from collections import defaultdict, Callable

from constants import USE_GPU
from constants import TASK_TYPE
from constants import TASK_LIST

from config import Configuration
from policy_network import PolicyNetwork
from train_dagger import DaggerThread

from imitation.discriminator import Discriminator
from imitation.network import Generator, Network
from imitation.gail import GailThread
from imitation.bc import BCThread
from imitation.dagger_mc import DaggerMCThread
from utils import nn

from scene_loader import THORDiscreteEnvironment as Environment
from expert import Expert


def change_target(scene_scope, target_cur, deviate_step):
    env = Environment({
        'scene_name': scene_scope,
    })
    env.current_state_id = target_cur
    expert = Expert(env)
    for _ in range(deviate_step):
        a = expert.get_next_action()
        env.step(a)
        env.update()
    target = env.current_state_id
    logging.info("changing target from %(target_cur)d to %(target)d" % locals())
    return target


def create_threads(config, model, network_scope, list_of_tasks, deviate_step=0):
    scene_scopes = list_of_tasks.keys()
    branches = [(scene, task) for scene in scene_scopes for task in list_of_tasks[scene]]
    if args.model == 'dagger':
        thread = DaggerThread
    elif args.model == 'gail':
        thread = GailThread
    elif args.model == 'bc':
        thread = BCThread
    elif args.model == 'dagger_mc':
        thread = DaggerMCThread
    else:
        raise ValueError("model not supported")

    threads = []
    for i, (scene_scope, task) in enumerate(branches):
        if deviate_step:
            task = str(change_target(scene_scope, int(task), deviate_step))
        threads.append(thread(config, model, i, network_scope=network_scope, scene_scope=scene_scope, task_scope=task))
    return threads


def get_logdir_str(config):
    keys = ('min_traj_per_train', 'max_iteration', 'policy_max_kl', 'lr', 'lr_vn', 'gan_d_cycle',
            'gan_v_cycle', 'lsr_epsilon', 'policy_ent_reg')
    return '-'.join([p+'_'+str(getattr(config, p)) for p in keys if hasattr(config, p)])


def train_model(model, session, config, threads, logdir, weight_root):
    if logdir:
        if not os.path.exists(logdir):
            os.makedirs(logdir)
        test_n = len(list(n for n in os.listdir(logdir) if n.startswith('t')))
        train_logdir = logdir + '/t' + str(test_n + 1) + '-' + args.model + '-' + get_logdir_str(config)
        logging.info("writing logs to %(train_logdir)s" % locals())
        summary_writer = tf.summary.FileWriter(train_logdir, session.graph)
    else:
        summary_writer = None

    if weight_root:
        if not os.path.exists(weight_root):
            os.makedirs(weight_root)
        weight_path = weight_root + '/modelv1'
        logging.info("loading from/writing weights to %(weight_path)s" % locals())
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
    train_start = time.time()
    lr_config = config.lr
    iteration = 0
    logging.info("total_threads=%d" % len(threads))
    while global_t < config.max_global_time_step and iteration < config.max_iteration:
        for thread in threads:
            if load_weights:
                thread.first_iteration = False
            global_t += thread.process(session, global_t, summary_writer)
            iteration += 1
            if saver and iteration % config.steps_per_save == (config.steps_per_save-1):
                logging.info('Save checkpoint at timestamp %d' % global_t)
                saver.save(session, weight_path)
            if iteration % config.steps_per_eval == (config.steps_per_eval-1):
                evaluate_model(session, config, thread.local_network, summary_writer, global_t)
            logging.debug("lr=%f" % thread.local_network.config.lr)
    duration = time.time() - train_start
    logging.info("global_t=%d, total_iterations=%d takes %0.2f s (%0.2f)" %
                 (global_t, iteration, duration/iteration, duration))
    if saver and not args.not_save_last:
        logging.info("Saving checkpoint at last")
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
        if args.model == 'dagger':
            model = PolicyNetwork(config, device=device, network_scope=network_scope, scene_scopes=scene_scopes)
        elif args.model in ('gail', 'bc', 'dagger_mc'):
            discriminator = Discriminator(config, scene_scopes=scene_scopes)
            generator = Generator(config, scene_scopes=scene_scopes)
            model = Network(config, generator, discriminator, scene_scopes=scene_scopes)
        else:
            raise ValueError("model not supported")
        sess_config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
        threads = create_threads(config, model, network_scope, list_of_tasks)
        with tf.Session(config=sess_config) as session:
            train_model(model, session, config, threads, config_dict['logdir'], config_dict['weight_root'])
        tf.reset_default_graph()


def train():
    config_dict = vars(args)
    train_models([config_dict]*args.max_attempt)


def search():
    config_dict = vars(args)
    # max_iteration = int(args.max_iteration)
    for _ in range(args.max_attempt):
        config_dict['lr'] = 10 ** np.random.uniform(-6, -3)
        # config_dict['lr'] = np.random.uniform(1.0e-5, 1.0e-4)
        # config_dict['lr_vn'] = 10 ** np.random.uniform(-5, -3)
        config_dict['lr_vn'] = np.random.uniform(2.4e-3, 2.5e-3)
        config_dict['policy_max_kl'] = np.random.uniform(1.0e-3, 1.0e-3)
        config_dict['wgan_lam'] = 10 ** np.random.uniform(-5, 0)
        config_dict['lsr_epsilon'] = np.random.uniform(1e-1, 1e-1)
        config_dict['policy_ent_reg'] = np.random.choice([1e-3])
        config_dict['min_traj_per_train'] = np.random.choice([10])
        # config_dict['max_iteration'] = int(config_dict['max_iteration'])*int(20.0/config_dict['min_traj_per_train'])
        t0 = time.time()
        train_models([config_dict])
        logging.info("training takes %.2f seconds" % (time.time() - t0))


def evaluate_model(session, config, model, summary_writer=None, global_step=0):
    network_scope = TASK_TYPE
    list_of_tasks = TASK_LIST
    threads = create_threads(config, model, network_scope, list_of_tasks, args.deviate_step)

    scene_stats = defaultdict(list)
    expert_stats = defaultdict(list)
    acc_stats = defaultdict(list)
    for thread in threads:
        scene = thread.scene_scope
        task = thread.task_scope
        logging.debug("evaluating " + scene + "/" + task)
        lengths, accuracies, exp_lengths, collisions, exp_collisions = \
            thread.evaluate(session, config.num_eval_episodes)
        logging.debug("Agent %s: mean_episode_length=%f/%f mean_episode_collision=%f/%f accuracies=%f" %
                      (scene+'/'+task, float(np.mean(lengths)),
                       float(np.mean(exp_lengths)),
                       float(np.mean(collisions)),
                       float(np.mean(exp_collisions)),
                       float(np.mean(accuracies))))
        scene_stats[scene].append(lengths)
        expert_stats[scene].append(exp_lengths)
        acc_stats[scene].append(accuracies)
    logging.info("Average_trajectory_length per scene @%d (steps):" % global_step)
    #  defined as traj length < 500
    for scene in scene_stats:
        logging.info("%s: agent=%f (acc=%.2f) expert=%f" %
                     (scene,
                      float(np.mean(scene_stats[scene])),
                      float(np.mean(acc_stats[scene])),
                      float(np.mean(expert_stats[scene])),
                      ))
        if summary_writer:
            summary_values = {
                "stats/" + scene + "-steps_eval": float(np.mean(scene_stats[scene])),
                "stats/" + scene + "-accuracy_eval": float(np.mean(acc_stats[scene])),
            }
            nn.add_summary(summary_writer, summary_values, global_step=global_step)
    if args.deviate_step:
        success_rate = np.mean((lengths<500).astype(float))
        logging.info("success rate is %(success_rate).3f" % locals())


def evaluate():
    device = "/gpu:0" if USE_GPU else "/cpu:0"
    network_scope = TASK_TYPE
    list_of_tasks = TASK_LIST
    scene_scopes = list_of_tasks.keys()
    weight_root = args.weight_root
    config = Configuration(**vars(args))
    print("evaluating with config=%s" % (str(config)))
    if args.model == 'dagger':
        model = PolicyNetwork(
            config, device=device, network_scope=network_scope, scene_scopes=scene_scopes)
    elif args.model in ('gail', 'bc', 'dagger_mc'):
        discriminator = Discriminator(config, scene_scopes=scene_scopes)
        generator = Generator(config, scene_scopes=scene_scopes)
        model = Network(config, generator, discriminator, scene_scopes=scene_scopes)
    else:
        raise ValueError("model not supported")
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
    parser.add_argument('--model', dest='model', default=None,
                        choices=('bc', 'gail', 'dagger', 'dagger_mc'))
    parser.add_argument('--max_attempt', dest='max_attempt', type=int, default=1,
                        help='search hyper parameters')
    parser.add_argument('--deviate_step', dest='deviate_step', default=0, type=int)
    parser.add_argument('--logdir', dest='logdir', default='./logdir', help='logdir')
    parser.add_argument('--weight_root', dest='weight_root', default=None, help='weights')
    parser.add_argument('--not_save_last', dest='not_save_last', default=False, action='store_true',
                        help='not save at last')
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
    t0 = time.time()
    if args.command:
        args.func()
        logging.info("%s takes %f s" % (args.command, time.time()-t0))
    else:
        parser.print_help()


