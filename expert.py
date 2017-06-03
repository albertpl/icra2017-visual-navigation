import argparse
import logging
import os
import time
import numpy as np
from collections import defaultdict, deque

from constants import TASK_LIST
from scene_loader import THORDiscreteEnvironment
from config import Configuration


class Expert(object):
    def __init__(self, env):
        self._actions = {
            0: '^',
            1: '>',
            2: '<',
            3: 'v',
        }
        self._env = env
        self._target = env.terminal_state_id
        self._pi, self._distance_matrix = self.build_sp(env.terminal_state_id)

    def verify_distance_matrix(self):
        return np.all(self._env.shortest_path_distances[self._target, :] == self._distance_matrix[self._target,:])

    def _reverse_action(self, a):
        return 3 - a

    def build_sp(self, target_id):
        """
        return a dictionary of tuples of actions for target_id and all_state id
        :param target_id:
        :return: dictionary
        """
        invalid_dist = 1000
        n_location = self._env.n_locations
        pi = {}
        # traverse all nodes
        q = deque([target_id])
        visit_map = np.zeros((n_location,))
        distance_matrix = np.ones((n_location, n_location), dtype=int) * invalid_dist
        distance_matrix[target_id, target_id] = 0
        visit_map[target_id] = True
        while len(q):
            t = q.pop()
            for a in self._actions.keys():
                n = self._env.transition_graph[t][a]
                if n != -1 and n not in pi and n != target_id:
                    assert not visit_map[n]
                    pi[n] = self._reverse_action(a)
                    distance_matrix[target_id, n] = distance_matrix[target_id, t] + 1
                    q.appendleft(n)
                    visit_map[n] = True
        # assert np.all(visit_map==True)
        return pi, distance_matrix

    def get_next_action(self):
        """
        return expert policy for target, current_state
        :param target_id:
        :param current_state:
        :return:
        """
        current_state=self._env.current_state_id
        assert self._target == self._env.terminal_state_id, "target id changes"
        assert current_state in self._pi, "unknown current state %d" % current_state
        a = self._pi[current_state]
        return a

    def get_a_str(self, a):
        return self._actions[a]


def test_scene(dump_file, test_cnt):
    config = Configuration()
    scene = os.path.basename(dump_file).split('.')[0]
    if scene not in TASK_LIST:
        env = THORDiscreteEnvironment({
            'h5_file_path': dump_file,
        })
        task_list = np.random.choice(list(range(env.n_locations)), test_cnt)
    else:
        task_list = TASK_LIST[scene]
    logging.info("testing scene %(scene)s task_list=%(task_list)s from dump file %(dump_file)s" % locals())

    for t in task_list:
        target = int(t)
        env = THORDiscreteEnvironment({
            'h5_file_path': dump_file,
            'terminal_state_id': target,
        })
        start = time.time()
        expert = Expert(env)
        logging.debug("building policy takes %f s" % (time.time()-start))
        assert expert.verify_distance_matrix()
        for _ in range(test_cnt):
            env.reset()
            logging.debug("scene=%s target=%d source=%d" % (scene, target, env.current_state_id))
            steps = []
            orig_state = env.current_state_id
            while not env.terminal:
                a = expert.get_next_action()
                logging.debug("state=%d action=%d" % (env.current_state_id, a))
                env.step(a)
                steps.append((env.current_state_id, a))
                assert len(steps) < config.max_steps_per_e, "current steps is beyond max_steps_per_e"
            logging.debug(str(orig_state)+''.join([expert.get_a_str(a)+str(s) for (s, a) in steps]))
            assert len(steps) == env.shortest_path_distances[orig_state][target]


def test():
    import glob
    for f in glob.glob(args.dump_root + '/*.h5'):
        test_scene(f, args.test_cnt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-l', dest='logLevel', default='info', help="logging level: {debug, info, error}")
    parser.add_argument("-c", "--test_cnt", type=int, default=1, help="# of checks per target")
    parser.add_argument("-s", "--dump_root", type=str, default="./data/", help="path to a hdf5 scene dumps")
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.logLevel.upper()),
                        format='%(asctime)s %(levelname)s %(message)s')

    test()
