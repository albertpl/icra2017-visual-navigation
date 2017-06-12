# -*- coding: utf-8 -*-

LOCAL_T_MAX = 5 # repeat step size
RMSP_ALPHA = 0.99 # decay parameter for RMSProp
RMSP_EPSILON = 0.1 # epsilon parameter for RMSProp
CHECKPOINT_DIR = 'checkpoints'
LOG_FILE = 'logs'
INITIAL_ALPHA_LOW = 1e-4    # log_uniform low limit for learning rate
INITIAL_ALPHA_HIGH = 1e-2   # log_uniform high limit for learning rate

PARALLEL_SIZE = 20 # parallel thread size
ACTION_SIZE = 4 # action size

INITIAL_ALPHA_LOG_RATE = 0.4226 # log_uniform interpolate rate for learning rate (around 7 * 10^-4)
GAMMA = 0.99 # discount factor for rewards
ENTROPY_BETA = 0.01 # entropy regurarlization constant
MAX_TIME_STEP = 10.0 * 10**6 # 10 million frames
GRAD_NORM_CLIP = 40.0 # gradient norm clipping
USE_GPU = True # To use GPU, set True
VERBOSE = True

SCREEN_WIDTH = 84
SCREEN_HEIGHT = 84
HISTORY_LENGTH = 4

NUM_EVAL_EPISODES = 100 # number of episodes for evaluation

TASK_TYPE = 'navigation' # no need to change
# keys are scene names, and values are a list of location ids (navigation targets)
if False:
    TASK_LIST = {
      'bathroom_02'    : ['26', '37', '43', '53', '69'],
      'bedroom_04'     : ['134', '264', '320', '384', '387'],
      'kitchen_02'     : ['90', '136', '157', '207', '329'],
      'living_room_08' : ['92', '135', '193', '228', '254']
    }
else:
    #   ','.join([" '" + str(d) + "'" for d in np.random.randint(1, 180, (10,))])
    TASK_LIST = {
        'bathroom_02': [  # 180
            '77', '79', '124', '126', '167', '175', '110', '51', '145', '177'
        ],
        'bedroom_04': [  #408
            '6', '237', '384', '222', '383', '8', '125', '101', '148', '52', '239', '177', '309', '89', '311', '70',
            '31', '119', '189', '123', '203', '391', '60', '173'
        ],
        'kitchen_02': [  #676
            '170', '154', '148', '406', '177', '161', '114', '121', '386', '78', '292', '403', '140', '1', '30', '236',
            '6', '32', '142', '260', '292', '86', '173', '57', '387', '96', '310', '359', '230', '262', '85', '81',
            '23', '25', '217', '175', '262', '153', '22', '395', '184'
        ],
        'living_room_08': [ # 468
            '167', '319', '191', '45', '380', '57', '216', '95', '52', '337', '41', '321', '146', '76', '142', '396',
            '139', '13', '63', '352', '381', '368', '170', '211', '67', '360', '198', '100', '245'
        ],
    }
    # TASK_LIST = {'bedroom_04': ['134']}
    # TASK_LIST = {'bathroom_02': ['26']}
    # TASK_LIST = {'living_room_08': ['92']}
    #TASK_LIST = {
        #'kitchen_02'     : ['90', '136', '157', '207', '329'],
    #    'kitchen_02'     : ['90'],
    #}
