import collections
from constants import LOCAL_T_MAX , MAX_TIME_STEP, NUM_EVAL_EPISODES

class Configuration(object):
    """Holds model hyper-parameters and data information.
    """
    # beta1 = 0.9  # Adam m_t <- beta1 * m_{t-1} + (1 - beta1) * g
    # beta2 = 0.999  # Adam v_t <- beta2 * v_{t-1} + (1 - beta2) * g * g
    action_size = 4
    batch_size = 128
    # dropout = 0.5
    early_stopping = 999
    gamma = 0.95  # discount factor
    local_t_max = LOCAL_T_MAX
    lr = 4.0e-4
    lr_vn = 1.0e-5   # lr for value network
    lr_decay_step = 1000  # decay by 0.9
    lsr_epsilon = 0.1
    relu_leakiness = 0.01  # Leaky Relu
    lam = 0.97  # GAE lambda
    max_epochs = 1   # of train per iteration
    max_global_time_step = MAX_TIME_STEP
    max_iteration = MAX_TIME_STEP  # maximum number of training
    max_steps_per_e = 5e3  # maximum steps per episode
    min_traj_per_train = 2
    num_eval_episodes = NUM_EVAL_EPISODES   # number of episodes per eval
    policy_max_kl = 0.01  # maximum KL divergency for TRPO
    policy_cg_damping = 0.01  # multiple of I to mix with Hessian for HVP
    reg = 0.04   # regularization for weight
    steps_per_save = 1e+3
    steps_per_eval = 1e+2

    def __init__(self, **kwargs):
        for key in [a for a in dir(self) if not isinstance(a, collections.Callable) and not a.startswith("__")]:
            if key in kwargs and kwargs[key] is not None:
                attr_type = type(getattr(self, key))
                v = attr_type(kwargs[key])
                setattr(self, key, v)

    def __str__(self):
        s = ''
        for key in sorted([a for a in dir(self) if not isinstance(a, collections.Callable) and not a.startswith("__")]):
            s += key + '-' + str(getattr(self, key)) + '_'
        return s
