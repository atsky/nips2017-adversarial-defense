import tensorflow as tf
from tensorflow.contrib import slim

import inception_resnet_v2
import functools

num_classes = 1001

is_training = False


def wrap_func(func):
    weight_decay = 0.0

    @functools.wraps(func)
    def network_fn(images, **kwargs):
        arg_scope = inception_resnet_v2.inception_resnet_v2_arg_scope(weight_decay=weight_decay)
        with slim.arg_scope(arg_scope):
            return func(images, num_classes, is_training=is_training, **kwargs)

    return network_fn
