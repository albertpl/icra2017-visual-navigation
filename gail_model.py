import tensorflow as tf
from tensorflow.contrib import layers
import math
import numpy as np

tf.logging.set_verbosity(tf.logging.INFO)


class Discriminator(object):
