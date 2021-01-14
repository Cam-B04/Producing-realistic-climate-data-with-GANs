# -*- coding: utf-8 -*-
"""
base models of Residual Attention Network
"""

import tensorflow as tf
from keras.utils import conv_utils
from keras import backend as K 
from keras.layers import InputSpec 
from keras.legacy import interfaces
from keras.utils.generic_utils import transpose_shape


class Layer(object):
    """basic layer"""
    def __init__(self, shape):
        """
        initial layer
        :param shape: shape of weight  (ex: [input_dim, output_dim]
        """
        # Xavier Initialization
        self.W = self.weight_variable(shape)
        self.b = tf.Variable(tf.zeros([shape[1]]))

    @staticmethod
    def weight_variable(shape, name=None):
        """define tensorflow variable"""
        # 標準偏差の2倍までのランダムな値で初期化
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial, name=name)

    def f_prop(self, x):
        """forward propagation"""
        return tf.matmul(x, self.W) + self.b


class Dense(Layer):
    """softmax layer """
    def __init__(self, shape, function=tf.nn.softmax):
        """
        :param shape: shape of weight (ex:[input_dim, output_dim]
        :param function: activation ex:)tf.nn.softmax
        """
        super().__init__(shape)
        # Xavier Initialization
        self.function = function

    def f_prop(self, x):
        """forward propagation"""
        return self.function(tf.matmul(x, self.W) + self.b)


class ResidualBlock(object):
    """
    residual block proposed by https://arxiv.org/pdf/1603.05027.pdf
    tensorflow version=r1.4
    """
    def __init__(self, kernel_size=3):
        """
        :param kernel_size: kernel size of second conv2d
        """
        self.kernel_size = kernel_size

    def f_prop(self, _input, input_channels, output_channels=None, scope="residual_block", is_training=True):
        """
        forward propagation
        :param _input: A Tensor
        :param input_channels: dimension of input channel.
        :param output_channels: dimension of output channel. input_channel -> output_channel
        :param stride: int stride of kernel
        :param scope: str, tensorflow name scope
        :param is_training: boolean, whether training step or not(test step)
        :return: output residual block
        """
        if output_channels is None:
            output_channels = input_channels

        with tf.variable_scope(scope):
            # batch normalization & ReLU TODO(this function should be updated when the TF version changes)
            x = self.batch_norm(_input, input_channels, is_training)

            x = tf.layers.conv2d(x, filters=output_channels, kernel_size=1, padding='SAME', name="conv1")

            # batch normalization & ReLU TODO(this function should be updated when the TF version changes)
            x = self.batch_norm(x, output_channels, is_training)

            x = tf.layers.conv2d(x, filters=output_channels, kernel_size=self.kernel_size,
                                 strides=1, padding='SAME', name="conv2")

            # update input
            if input_channels != output_channels:
                _input = tf.layers.conv2d(_input, filters=output_channels, kernel_size=1, strides=1)

            output = x + _input

            return output

    @staticmethod
    def batch_norm(x, n_out, is_training=True):
        """
        Batch normalization on convolutional maps.
        Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
        Args:
            x:           Tensor, 4D BHWD input maps
            n_out:       integer, depth of input maps
            is_training: boolean tf.Varialbe, true indicates training phase
            scope:       string, variable scope
        Return:
            normed:      batch-normalized maps
        """
        with tf.variable_scope('batch_norm'):
            beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                               name='beta', trainable=True)
            gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                                name='gamma', trainable=True)
            batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
            ema = tf.train.ExponentialMovingAverage(decay=0.5)

            def mean_var_with_update():
                ema_apply_op = ema.apply([batch_mean, batch_var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(batch_mean), tf.identity(batch_var)

            mean, var = tf.cond(tf.cast(is_training, tf.bool),
                                mean_var_with_update,
                                lambda: (ema.average(batch_mean), ema.average(batch_var)))
            normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
        return tf.nn.relu(normed)

class _UpSampling(Layer):
    """Abstract nD UpSampling layer (private, used as implementation base).
    # Arguments
        size: Tuple of ints.
        data_format: A string,
            one of `"channels_last"` or `"channels_first"`.
            The ordering of the dimensions in the inputs.
            `"channels_last"` corresponds to inputs with shape
            `(batch, ..., channels)` while `"channels_first"` corresponds to
            inputs with shape `(batch, channels, ...)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".
    """
    def __init__(self, size, data_format=None, **kwargs):
        # self.rank is 1 for UpSampling1D, 2 for UpSampling2D.
        self.rank = len(size)
        self.size = size
        self.data_format = K.normalize_data_format(data_format)
        self.input_spec = InputSpec(ndim=self.rank + 2)
        super(_UpSampling, self).__init__(**kwargs)

    def call(self, inputs):
        raise NotImplementedError

    def compute_output_shape(self, input_shape):
        size_all_dims = (1,) + self.size + (1,)
        spatial_axes = list(range(1, 1 + self.rank))
        size_all_dims = transpose_shape(size_all_dims,
                                        self.data_format,
                                        spatial_axes)
        output_shape = list(input_shape)
        for dim in range(len(output_shape)):
            if output_shape[dim] is not None:
                output_shape[dim] *= size_all_dims[dim]
        return tuple(output_shape)

    def get_config(self):
        config = {'size': self.size,
                  'data_format': self.data_format}
        base_config = super(_UpSampling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class UpSampling2D(_UpSampling):
    """Upsampling layer for 2D inputs.
    Repeats the rows and columns of the data
    by size[0] and size[1] respectively.
    # Arguments
        size: int, or tuple of 2 integers.
            The upsampling factors for rows and columns.
        data_format: A string,
            one of `"channels_last"` or `"channels_first"`.
            The ordering of the dimensions in the inputs.
            `"channels_last"` corresponds to inputs with shape
            `(batch, height, width, channels)` while `"channels_first"`
            corresponds to inputs with shape
            `(batch, channels, height, width)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".
        interpolation: A string, one of `nearest` or `bilinear`.
            Note that CNTK does not support yet the `bilinear` upscaling
            and that with Theano, only `size=(2, 2)` is possible.
    # Input shape
        4D tensor with shape:
        - If `data_format` is `"channels_last"`:
            `(batch, rows, cols, channels)`
        - If `data_format` is `"channels_first"`:
            `(batch, channels, rows, cols)`
    # Output shape
        4D tensor with shape:
        - If `data_format` is `"channels_last"`:
            `(batch, upsampled_rows, upsampled_cols, channels)`
        - If `data_format` is `"channels_first"`:
            `(batch, channels, upsampled_rows, upsampled_cols)`
    """

    #@interfaces.legacy_upsampling2d_support
    def __init__(self, size=(2, 2), data_format=None, interpolation='nearest',
                 **kwargs):
        normalized_size = conv_utils.normalize_tuple(size, 2, 'size')
        super(UpSampling2D, self).__init__(normalized_size, data_format, **kwargs)
        if interpolation not in ['nearest', 'bilinear']:
            raise ValueError('interpolation should be one '
                             'of "nearest" or "bilinear".')
        self.interpolation = interpolation

    def call(self, inputs):
        return K.resize_images(inputs, self.size[0], self.size[1],
                               self.data_format, self.interpolation)

    def get_config(self):
        config = super(UpSampling2D, self).get_config()
        config['interpolation'] = self.interpolation
        return config

    @property
    def outbound_nodes(self):
        if hasattr(self, '_outbound_nodes'):
            print("outbound_nodes called but _outbound_nodes found")
        return getattr(self, '_outbound_nodes', [])


