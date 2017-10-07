from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import base
from tensorflow.python.layers import utils
import tensorflow as tf

from tensorflow.python.ops import variables as tf_variables


class ConvBayes(base.Layer):
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=1,
                 padding='valid',
                 data_format='channels_last',
                 dilation_rate=1,
                 activation=None,
                 use_bias=True,
                 kernel_initializer=None,
                 bias_initializer=init_ops.zeros_initializer(),
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 trainable=True,
                 name=None,
                 **kwargs):
        super(ConvBayes, self).__init__(trainable=trainable,
                                        name=name, **kwargs)
        rank = 2
        self.rank = rank
        self.filters = filters
        self.kernel_size = utils.normalize_tuple(kernel_size, rank, 'kernel_size')
        self.strides = utils.normalize_tuple(strides, rank, 'strides')
        self.padding = utils.normalize_padding(padding)
        self.data_format = utils.normalize_data_format(data_format)
        self.dilation_rate = utils.normalize_tuple(
            dilation_rate, rank, 'dilation_rate')
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = None
        self.bias_regularizer = bias_regularizer
        self.activity_regularizer = activity_regularizer
        self.input_spec = base.InputSpec(ndim=self.rank + 2)

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis].value is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = input_shape[channel_axis].value
        kernel_shape = self.kernel_size + (input_dim, self.filters)

        #kernel_size = kernel_shape[0] * kernel_shape[1] * kernel_shape[2] * kernel_shape[3]

        self.local_approximation = False  # kernel_size > 200000
        # print(kernel_shape, kernel_size, self.local_approximation)

        existing_variables = set(tf_variables.global_variables())

        self.kernel = self.add_variable(name='kernel',
                                        shape=kernel_shape,
                                        initializer=self.kernel_initializer,
                                        regularizer=None,
                                        trainable=True,
                                        dtype=self.dtype)


        self.sqrt_sd_var = self.add_variable(name='sqrt_sd',
                                    shape=kernel_shape,
                                    initializer=tf.constant_initializer(1),
                                    regularizer=None,
                                    trainable=True,
                                    dtype=self.dtype)

        self.sd = tf.abs(self.sqrt_sd_var) * 0.1 + 1e-10

        if self.sqrt_sd_var not in existing_variables:
            log_sd = tf.log(self.sd)

            tf.add_to_collection("log_sd", log_sd)

            prior_sd_2 = 0.01

            tf.add_to_collection("update_sd", tf.assign(self.sqrt_sd_var, tf.abs(self.kernel) * 0.5))

            kl = tf.reduce_sum((self.kernel ** 2 + tf.square(self.sd)) * (0.5 / prior_sd_2) - log_sd)
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.identity(2e-9 * kl, name="kl-loss"))

        if self.use_bias:
            self.bias = self.add_variable(name='bias',
                                          shape=(self.filters,),
                                          initializer=self.bias_initializer,
                                          regularizer=self.bias_regularizer,
                                          trainable=True,
                                          dtype=self.dtype)
        else:
            self.bias = None
        self.input_spec = base.InputSpec(ndim=self.rank + 2,
                                         axes={channel_axis: input_dim})
        self.built = True

    def call(self, inputs):
        if self.local_approximation:
            outputs_mean = nn.conv2d(
                input=inputs,
                filter=self.kernel,
                strides=[1] + list(self.strides) + [1],
                padding=self.padding.upper(),
                data_format=utils.convert_data_format(self.data_format, self.rank + 2))

            outputs_var = nn.conv2d(
                input=tf.square(inputs),
                filter=tf.square(self.sd),
                strides=[1] + list(self.strides) + [1],
                padding=self.padding.upper(),
                data_format=utils.convert_data_format(self.data_format, self.rank + 2))

            norm = tf.random_normal(tf.shape(outputs_mean), mean=0, stddev=1)

            outputs = outputs_mean + tf.sqrt(outputs_var) * norm
        else:
            norm = tf.random_normal(tf.shape(self.kernel), mean=0, stddev=1)

            W = self.kernel + self.sd * norm

            outputs = nn.conv2d(
                input=inputs,
                filter=W,
                strides=[1] + list(self.strides) + [1],
                padding=self.padding.upper(),
                data_format=utils.convert_data_format(self.data_format, self.rank + 2))

        if self.bias is not None:
            if self.data_format == 'channels_first':
                # bias_add only supports NHWC.
                # TODO(fchollet): remove this when `bias_add` is feature-complete.
                if self.rank == 1:
                    bias = array_ops.reshape(self.bias, (1, self.filters, 1))
                    outputs += bias
                if self.rank == 2:
                    bias = array_ops.reshape(self.bias, (1, self.filters, 1, 1))
                    outputs += bias
                if self.rank == 3:
                    # As of Mar 2017, direct addition is significantly slower than
                    # bias_add when computing gradients. To use bias_add, we collapse Z
                    # and Y into a single dimension to obtain a 4D input tensor.
                    outputs_shape = outputs.shape.as_list()
                    outputs_4d = array_ops.reshape(outputs,
                                                   [outputs_shape[0], outputs_shape[1],
                                                    outputs_shape[2] * outputs_shape[3],
                                                    outputs_shape[4]])
                    outputs_4d = nn.bias_add(outputs_4d, self.bias, data_format='NCHW')
                    outputs = array_ops.reshape(outputs_4d, outputs_shape)
            else:
                outputs = nn.bias_add(outputs, self.bias, data_format='NHWC')

        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def _compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        if self.data_format == 'channels_last':
            space = input_shape[1:-1]
            new_space = []
            for i in range(len(space)):
                new_dim = utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            return tensor_shape.TensorShape([input_shape[0]] + new_space +
                                            [self.filters])
        else:
            space = input_shape[2:]
            new_space = []
            for i in range(len(space)):
                new_dim = utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            return tensor_shape.TensorShape([input_shape[0], self.filters] +
                                            new_space)
