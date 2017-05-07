import collections
from constants import MAX_STEPS_PER_E
from constants import MAX_TIME_STEP

class Configuration(object):
    """Holds model hyper-parameters and data information.
    """
    # beta1 = 0.9  # Adam m_t <- beta1 * m_{t-1} + (1 - beta1) * g
    # beta2 = 0.999  # Adam v_t <- beta2 * v_{t-1} + (1 - beta2) * g * g
    action_size = 4
    batch_size = 32
    # dropout = 0.5
    early_stopping = 999
    lr = 1e-3
    relu_leakiness = 0.01 # Leaky Relu
    max_epochs = 1   # # of train per iteration
    max_global_time_step = MAX_TIME_STEP
    max_steps_per_e = MAX_STEPS_PER_E
    reg = 0.04   # regularization for weight

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
