import tensorflow as tf
import keras.backend as K
from keras import activations, initializers, regularizers, constraints
from keras.utils import conv_utils
from keras.utils.generic_utils import transpose_shape
from keras.layers import InputSpec, MaxPooling2D, AveragePooling2D
from keras.layers import Input, Add, Layer, Dense, Conv2D, BatchNormalization
from keras.layers import Flatten, Conv2DTranspose, Activation
from keras.layers import Embedding, concatenate, Conv1D, Conv3D
from keras.layers.pooling import _GlobalPooling2D
from keras.models import Model
from keras.engine import *
from keras.legacy import interfaces
from src.modeling.basic_layer import UpSampling2D
import numpy as np


def periodic_padding(image, padding):
    """
    Create a periodic padding (wrap) around the image, to emulate periodic
    boundary conditions
    """
    # print(padding)
    # pad_h = tf.cast(padding[0][0], tf.int32)
    # pad_w = tf.cast(padding[1][1], tf.int32)
    assert len(padding) == 2
    assert len(padding[0]) == 2
    assert len(padding[1]) == 2
    # data_format = normalize_data_format(data_format)

    pad_h = padding[0][0]
    pad_w = padding[1][1]
    padding = (pad_h, pad_w)
    if padding[0] != 0:
        upper_pad = image[:, -1 * padding[0]:, :]
        lower_pad = image[:, : padding[0], :]
        padded_image = tf.concat([upper_pad, image, lower_pad], axis=1)

    elif padding[1] != 0:
        left_pad = image[:, :, -1 * padding[1]:]
        right_pad = image[:, :, : padding[1]]
        padded_image = tf.concat([left_pad, image, right_pad], axis=2)

    return padded_image


def periodic_paddingv2(image, padding=1):
    """
    Create a periodic padding (wrap) around the image, to emulate periodic
    boundary conditions
    """

    rows, columns = image.shape

    # create left matrix
    left_corner_diagonal = tf.eye(padding)
    left_filled_zeros = tf.zeros([padding, rows.value - padding])

    left_upper = tf.concat([left_filled_zeros, left_corner_diagonal], axis=1)
    left_center_diagonal = tf.eye(rows.value)
    left_lower = tf.concat([left_corner_diagonal, left_filled_zeros], axis=1)

    left_matrix = tf.concat([left_upper, left_center_diagonal, left_lower],
                            axis=0)

    # create right matrix
    right_corner_diagonal = tf.eye(padding)
    right_filled_zeros = tf.zeros([columns.value - padding, padding])

    right_left_side = tf.concat([right_filled_zeros, right_corner_diagonal],
                                axis=0)
    right_center_diagonal = tf.eye(columns.value)
    right_right_side = tf.concat([right_corner_diagonal, right_filled_zeros],
                                 axis=0)

    right_matrix = tf.concat(
        [right_left_side, right_center_diagonal, right_right_side], axis=1
    )

    # left and right matrices are immutable
    return tf.matmul(left_matrix, tf.matmul(image, right_matrix))


def nearest_padding(image, padding):
    """
    Create a periodic padding (wrap) around the image, to emulate periodic
    boundary conditions
    """
    pad_h = padding[0][0]
    pad_w = padding[1][1]
    padding = (pad_h, pad_w)

    if padding[0] != 0:
        upper_pad = image[:, :1, :]
        lower_pad = image[:, -1:, :]
        L = [upper_pad, image, lower_pad]
        for i in range(padding[0] - 1):
            L.insert(0, upper_pad)
            L.insert(-1, lower_pad)

        padded_image = tf.concat(L, axis=1)

    elif padding[1] != 0:

        left_pad = image[:, :, :1]
        right_pad = image[:, :, -1:]
        for i in range(padding[0] - 1):
            L.insert(0, left_pad)
            L.insert(-1, right_pad)

            padded_image = tf.concat(L, axis=2)

    return padded_image


class _ZeroPadding(Layer):
    """Abstract nD ZeroPadding layer (private, used as implementation base).
    # Arguments
        padding: Tuple of tuples of two ints. Can be a tuple of ints when
            rank is 1.
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

    def __init__(self, padding, data_format="channels_last", **kwargs):
        # self.rank is 1 for ZeroPadding1D, 2 for ZeroPadding2D.
        self.rank = len(padding)
        self.padding = padding
        self.data_format = K.normalize_data_format(data_format)
        self.input_spec = InputSpec(ndim=self.rank + 2)
        super(_ZeroPadding, self).__init__(**kwargs)

    def call(self, inputs):
        raise NotImplementedError

    def compute_output_shape(self, input_shape):
        padding_all_dims = ((0, 0),) + self.padding + ((0, 0),)
        spatial_axes = list(range(1, 1 + self.rank))
        padding_all_dims = transpose_shape(
            padding_all_dims, self.data_format, spatial_axes
        )
        output_shape = list(input_shape)
        for dim in range(len(output_shape)):
            if output_shape[dim] is not None:
                output_shape[dim] += sum(padding_all_dims[dim])
        return tuple(output_shape)

    def get_config(self):
        config = {"padding": self.padding, "data_format": self.data_format}
        base_config = super(_ZeroPadding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class WrapPadding2D(_ZeroPadding):
    """Zero-padding layer for 2D input (e.g. picture).
    This layer can add rows and columns of zeros
    at the top, bottom, left and right side of an image tensor.
    # Arguments
        padding: int, or tuple of 2 ints, or tuple of 2 tuples of 2 ints.
            - If int: the same symmetric padding
                is applied to height and width.
            - If tuple of 2 ints:
                interpreted as two different
                symmetric padding values for height and width:
                `(symmetric_height_pad, symmetric_width_pad)`.
            - If tuple of 2 tuples of 2 ints:
                interpreted as
                `((top_pad, bottom_pad), (left_pad, right_pad))`
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
    # Input shape
        4D tensor with shape:
        - If `data_format` is `"channels_last"`:
            `(batch, rows, cols, channels)`
        - If `data_format` is `"channels_first"`:
            `(batch, channels, rows, cols)`
    # Output shape
        4D tensor with shape:
        - If `data_format` is `"channels_last"`:
            `(batch, padded_rows, padded_cols, channels)`
        - If `data_format` is `"channels_first"`:
            `(batch, channels, padded_rows, padded_cols)`
    """

    # @interfaces.legacy_zeropadding2d_support
    def __init__(self, padding=(1, 1), data_format="channels_last", **kwargs):
        # self.padding = padding
        if isinstance(padding, int):
            normalized_padding = ((padding, padding), (padding, padding))
        elif hasattr(padding, "__len__"):
            if len(padding) != 2:
                raise ValueError(
                    f"`padding` should have two elements. Found: {padding}"
                )
            height_padding = conv_utils.normalize_tuple(
                padding[0], 2, "1st entry of padding"
            )
            width_padding = conv_utils.normalize_tuple(
                padding[1], 2, "2nd entry of padding"
            )
            normalized_padding = (height_padding, width_padding)
        else:
            raise ValueError(
                "`padding` should be either an int, "
                "a tuple of 2 ints "
                "(symmetric_height_pad, symmetric_width_pad), "
                "or a tuple of 2 tuples of 2 ints "
                "((top_pad, bottom_pad), (left_pad, right_pad)). "
                "Found: " + str(padding)
            )
        super(WrapPadding2D, self).__init__(normalized_padding, data_format,
                                            **kwargs)

    def call(self, inputs):
        return periodic_padding(inputs, padding=self.padding)


class NearestPadding2D(_ZeroPadding):
    """Zero-padding layer for 2D input (e.g. picture).
    This layer can add rows and columns of zeros
    at the top, bottom, left and right side of an image tensor.
    # Arguments
        padding: int, or tuple of 2 ints, or tuple of 2 tuples of 2 ints.
            - If int: the same symmetric padding
                is applied to height and width.
            - If tuple of 2 ints:
                interpreted as two different
                symmetric padding values for height and width:
                `(symmetric_height_pad, symmetric_width_pad)`.
            - If tuple of 2 tuples of 2 ints:
                interpreted as
                `((top_pad, bottom_pad), (left_pad, right_pad))`
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
    # Input shape
        4D tensor with shape:
        - If `data_format` is `"channels_last"`:
            `(batch, rows, cols, channels)`
        - If `data_format` is `"channels_first"`:
            `(batch, channels, rows, cols)`
    # Output shape
        4D tensor with shape:
        - If `data_format` is `"channels_last"`:
            `(batch, padded_rows, padded_cols, channels)`
        - If `data_format` is `"channels_first"`:
            `(batch, channels, padded_rows, padded_cols)`
    """

    @interfaces.legacy_zeropadding2d_support
    def __init__(self, padding=(1, 1), data_format=None, **kwargs):

        if isinstance(padding, int):
            normalized_padding = ((padding, padding), (padding, padding))
        elif hasattr(padding, "__len__"):
            if len(padding) != 2:
                raise ValueError(
                    f"`padding` should have two elements. {padding}"
                )
            height_padding = conv_utils.normalize_tuple(
                padding[0], 2, "1st entry of padding"
            )
            width_padding = conv_utils.normalize_tuple(
                padding[1], 2, "2nd entry of padding"
            )
            normalized_padding = (height_padding, width_padding)
        else:
            raise ValueError(
                "`padding` should be either an int, "
                "a tuple of 2 ints "
                "(symmetric_height_pad, symmetric_width_pad), "
                "or a tuple of 2 tuples of 2 ints "
                "((top_pad, bottom_pad), (left_pad, right_pad)). "
                "Found: " + str(padding)
            )
        super(NearestPadding2D, self).__init__(
            normalized_padding, data_format, **kwargs
        )

    def call(self, inputs):
        return nearest_padding(inputs, padding=self.padding)


class PB_layer_pad(Layer):
    def __init__(self, img_shape, axis, padding, **kwargs):
        # self.output_dim = output_dim
        self.img_rows = img_shape[0]
        self.img_cols = img_shape[1]
        self.channels = img_shape[2]
        self.padding = padding
        self.axis = axis
        super(PB_layer_pad, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(
            name="kernel", shape=(1, 1), initializer="uniform", trainable=False
        )
        # Be sure to call this at the end
        super(PB_layer_pad, self).build(input_shape)

    def call(self, tensor):
        import tensorflow as tf
        import keras.backend as K

        """
        add periodic padding to a tensor for specified axis
        tensor: input tensor
        axis: on or multiple axis to pad along, int or tuple
        padding: number of cells to pad, int or tuple

        return: padded tensor
        """

        if isinstance(self.axis, int):
            self.axis_ = (self.axis,)
        if isinstance(self.padding, int):
            self.padding_ = (self.padding,)

        ndim = 4
        for ax, p in zip(self.axis_, self.padding_):
            # create a slice object that selects everything from all axes,
            # except only 0:p for the specified for right, and -p: for left

            ind_right = [slice(-p, None) if i == ax else slice(None)
                         for i in range(ndim)]
            ind_left = [slice(0, p) if i == ax else slice(None)
                        for i in range(ndim)]
            right = tensor[ind_right]
            print(ind_right)
            left = tensor[ind_left]
            middle = tensor
            tensor = tf.concat([right, middle, left], axis=ax)
        return tensor

    def compute_output_shape(self, input_shape):
        if self.axis_[0] == 1:
            # shape = (None,64,132,5)
            shape = (
                input_shape[0],
                input_shape[1] + 2 * self.padding,
                input_shape[2],
                input_shape[3],
            )
        elif self.axis_[0] == 2:
            # shape = (None,64,132,5)
            shape = (
                input_shape[0],
                input_shape[1],
                input_shape[2] + 2 * self.padding,
                input_shape[3],
            )

        # else:
        # shape = (input_shape[0], input_shape[1], input_shape[2]+self.padding,
        #  input_shape[3])
        return shape


def identity_block(X, f, filters, stage, block):

    # Defining name basis
    conv_name_base = f'res{stage}{block}_branch'
    bn_name_base = f'bn{stage}{block}_branch'

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value
    X_shortcut = X

    # First component of main path
    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(1, 1),
               padding='valid', name=f'{conv_name_base}2a',
               kernel_initializer='glorot_uniform')(X)
    X = Activation('relu')(X)

    # Second component of main path
    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1),
               padding='same', name=f'{conv_name_base}2b',
               kernel_initializer='glorot_uniform')(X)
    X = Activation('relu')(X)

    # Third component of main path
    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1),
               padding='valid', name=f'{conv_name_base}2c',
               kernel_initializer='glorot_uniform')(X)

    # Final step: Add shortcut value to main path, and pass it through
    # a RELU activation
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X


def convolutional_block(X, f, filters, stage, block, s=2):

    # Defining name basis
    conv_name_base = f"res{stage}{block}_branch"
    bn_name_base = f"bn{stage}{block}_branch"

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value
    X_shortcut = X

    # #### MAIN PATH #####
    # First component of main path
    X = Conv2D(
        filters=F1,
        kernel_size=(1, 1),
        strides=(s, s),
        padding="valid",
        name=f'{conv_name_base}2a',
        kernel_initializer='glorot_uniform',
    )(X)
    X = Activation("relu")(X)

    # Second component of main path
    X = Conv2D(
        filters=F2,
        kernel_size=(f, f),
        strides=(1, 1),
        padding="same",
        name=f'{conv_name_base}2b',
        kernel_initializer='glorot_uniform',
    )(X)
    X = Activation("relu")(X)

    # Third component of main path
    X = Conv2D(
        filters=F3,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding="valid",
        name=f'{conv_name_base}2c',
        kernel_initializer='glorot_uniform',
    )(X)

    # #### SHORTCUT PATH ####
    X_shortcut = Conv2D(
        filters=F3,
        kernel_size=(1, 1),
        strides=(s, s),
        padding="valid",
        name=conv_name_base + "1",
        kernel_initializer='glorot_uniform',
    )(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU
    # activation
    X = Add()([X, X_shortcut])
    X = Activation("relu")(X)

    return X


def ResNet50(input_shape=(64, 64, 3), classes=6, warp=True):

    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)

    # WarpPadding
    if warp:
        X = WrapPadding2D(padding=(0, 6))(X_input)
    else:
        X = Conv2D(
            81,
            (7, 7),
            strides=(1, 1),
            name="conv1",
            kernel_initializer='glorot_uniform',
        )(X_input)
    # Stage 1
    X = Conv2D(
        81,
        (7, 7),
        strides=(2, 2),
        name="conv1",
        kernel_initializer='glorot_uniform',
    )(X)
    # X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)
    X = Activation("relu")(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 2
    X = convolutional_block(X, f=3, filters=[64, 64, 128], stage=2,
                            block="a", s=2)
    X = identity_block(X, 3, [81, 81, 128], stage=2, block="b")
    X = identity_block(X, 3, [81, 81, 128], stage=2, block="c")

    # Stage 3
    X = convolutional_block(X, f=3, filters=[128, 128, 128], stage=3,
                            block="a", s=2)
    X = identity_block(X, 3, [128, 128, 256], stage=3, block="b")
    X = identity_block(X, 3, [128, 128, 256], stage=3, block="c")
    X = identity_block(X, 3, [128, 128, 256], stage=3, block="d")

    # Stage 4
    X = convolutional_block(X, f=3, filters=[128, 128, 256], stage=4,
                            block="a", s=2)
    X = identity_block(X, 3, [256, 256, 256], stage=4, block="b")
    X = identity_block(X, 3, [256, 256, 256], stage=4, block="c")
    X = identity_block(X, 3, [256, 256, 256], stage=4, block="d")
    X = identity_block(X, 3, [256, 256, 256], stage=4, block="e")
    # X = identity_block(X, 3, [128, 128, 256], stage=4, block='f')

    # AVGPOOL
    X = AveragePooling2D(pool_size=(2, 2), padding="same")(X)

    # Output layer
    X = Flatten()(X)
    X = Dense(1, kernel_initializer='glorot_uniform')(X)

    # Create model
    return Model(inputs=X_input, outputs=X, name="ResNet50")


def inv_identity_block(X, f, filters, stage, block):

    # Defining name basis
    conv_name_base = f"res{stage}{block}_branch"
    bn_name_base = f"bn{stage}{block}_branch"

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value
    X_shortcut = X

    # First component of main path
    X = Conv2DTranspose(
        filters=F1,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding="valid",
        name="{conv_name_base}2a",
        kernel_initializer='glorot_uniform',
    )(X)

    X = Activation("relu")(X)

    # Second component of main path
    X = Conv2DTranspose(
        filters=F2,
        kernel_size=(f, f),
        strides=(1, 1),
        padding="same",
        name=f"{conv_name_base}2b",
        kernel_initializer='glorot_uniform',
    )(X)

    X = Activation("relu")(X)

    # Third component of main path
    X = Conv2DTranspose(
        filters=F3,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding="valid",
        name=f"{conv_name_base}2c",
        kernel_initializer='glorot_uniform',
    )(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU
    # activation
    X = Add()([X, X_shortcut])
    X = Activation("relu")(X)

    return X


def inv_convolutional_block(X, f, filters, stage, block, s=2):

    # Defining name basis
    conv_name_base = f"res{stage}{block}_branch"
    bn_name_base = f"bn{stage}{block}_branch"

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value
    X_shortcut = X

    # #### MAIN PATH #####
    # First component of main path
    X = Conv2DTranspose(
        filters=F1,
        kernel_size=(1, 1),
        strides=(s, s),
        output_padding=1,
        padding="valid",
        name=f"{conv_name_base}2a",
        kernel_initializer='glorot_uniform',
    )(X)

    X = Activation("relu")(X)

    # Second component of main path
    X = Conv2DTranspose(
        filters=F2,
        kernel_size=(f, f),
        strides=(1, 1),
        padding="same",
        name=f"{conv_name_base}2b",
        kernel_initializer='glorot_uniform',
    )(X)

    X = Activation("relu")(X)

    # Third component of main path
    X = Conv2DTranspose(
        filters=F3,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding="valid",
        name=f"{conv_name_base}2c",
        kernel_initializer='glorot_uniform',
    )(X)

    # #### SHORTCUT PATH ####
    X_shortcut = Conv2DTranspose(
        filters=F3,
        kernel_size=(1, 1),
        output_padding=1,
        strides=(s, s),
        padding="valid",
        name=f"{conv_name_base}1",
        kernel_initializer='glorot_uniform',
    )(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU
    # activation
    X = Add()([X, X_shortcut])
    X = Activation("relu")(X)

    return X


def compact_bilinear(tensors_list):
    def _generate_sketch_matrix(rand_h, rand_s, output_dim):
        """
        Return a sparse matrix used for tensor sketch operation in compact
        bilinear pooling
        Args:
            rand_h: an 1D numpy array containing indices in interval
            `[0, output_dim)`. rand_s: an 1D numpy array of 1 and -1, having
            the same shape as `rand_h`.
            output_dim: the output dimensions of compact bilinear pooling.
        Returns:
            a sparse matrix of shape [input_dim, output_dim] for tensor sketch.
        """

        # Generate a sparse matrix for tensor count sketch
        rand_h = rand_h.astype(np.int64)
        rand_s = rand_s.astype(np.float32)
        assert (rand_h.ndim == 1 and
                rand_s.ndim == 1 and
                len(rand_h) == len(rand_s))
        assert np.all(rand_h >= 0) and np.all(rand_h < output_dim)

        input_dim = len(rand_h)
        indices = np.concatenate(
            (np.arange(input_dim)[..., np.newaxis], rand_h[..., np.newaxis]),
            axis=1)

        return tf.sparse_reorder(
            tf.SparseTensor(indices, rand_s, [input_dim, output_dim]))
    bottom1, bottom2 = tensors_list
    output_dim = 8192

    # Static shapes are needed to construction count sketch matrix
    input_dim1 = bottom1.get_shape().as_list()[-1]
    input_dim2 = bottom2.get_shape().as_list()[-1]

    # print (bottom1.get_shape().as_list())
    # print (bottom2.get_shape().as_list())

    # Step 0: Generate vectors and sketch matrix for tensor count sketch
    # This is only done once during graph construction, and fixed during each
    # operation
    seed_h_1 = 1
    seed_s_1 = 3
    seed_h_2 = 5
    seed_s_2 = 7

    # Generate sparse_sketch_matrix1 using rand_h_1 and rand_s_1
    np.random.seed(seed_h_1)
    rand_h_1 = np.random.randint(output_dim, size=input_dim1)
    np.random.seed(seed_s_1)
    rand_s_1 = 2 * np.random.randint(2, size=input_dim1) - 1
    sparse_sketch_matrix1 = _generate_sketch_matrix(rand_h_1, rand_s_1,
                                                    output_dim)

    # Generate sparse_sketch_matrix2 using rand_h_2 and rand_s_2
    np.random.seed(seed_h_2)
    rand_h_2 = np.random.randint(output_dim, size=input_dim2)
    np.random.seed(seed_s_2)
    rand_s_2 = 2 * np.random.randint(2, size=input_dim2) - 1
    sparse_sketch_matrix2 = _generate_sketch_matrix(rand_h_2, rand_s_2,
                                                    output_dim)

    # Step 1: Flatten the input tensors and count sketch
    bottom1_flat = tf.reshape(bottom1, [-1, input_dim1])
    bottom2_flat = tf.reshape(bottom2, [-1, input_dim2])

    # Essentially:
    #   sketch1 = bottom1 * sparse_sketch_matrix
    #   sketch2 = bottom2 * sparse_sketch_matrix
    # But tensorflow only supports left multiplying a sparse matrix, so:
    #   sketch1 = (sparse_sketch_matrix.T * bottom1.T).T
    #   sketch2 = (sparse_sketch_matrix.T * bottom2.T).T
    sketch1 = tf.transpose(
        tf.sparse_tensor_dense_matmul(
            sparse_sketch_matrix1, bottom1_flat, adjoint_a=True, adjoint_b=True
        )
    )
    sketch2 = tf.transpose(
        tf.sparse_tensor_dense_matmul(
            sparse_sketch_matrix2, bottom2_flat, adjoint_a=True, adjoint_b=True
        )
    )

    # Step 2: FFT
    fft1 = tf.fft(tf.complex(real=sketch1, imag=tf.zeros_like(sketch1)))
    fft2 = tf.fft(tf.complex(real=sketch2, imag=tf.zeros_like(sketch2)))

    # Step 3: Elementwise product
    fft_product = tf.multiply(fft1, fft2)

    # Step 4: Inverse FFT and reshape back
    # Compute output shape dynamically: [batch_size, height, width, output_dim]
    cbp_flat = tf.real(tf.ifft(fft_product))

    output_shape = tf.add(
        tf.multiply(tf.shape(bottom1), [1, 1, 1, 0]), [0, 0, 0, output_dim]
    )
    cbp = tf.reshape(cbp_flat, output_shape)

    # print (cbp.get_shape().as_list())

    return cbp


class DenseSN(Dense):
    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]
        self.kernel = self.add_weight(
            shape=(input_dim, self.units),
            initializer=self.kernel_initializer,
            name="kernel",
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.units,),
                initializer=self.bias_initializer,
                name="bias",
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
            )
        else:
            self.bias = None
        self.u = self.add_weight(
            shape=(1, self.kernel.shape.as_list()[-1]),
            initializer=initializers.RandomNormal(0, 1),
            name="sn",
            trainable=False,
        )
        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def call(self, inputs, training=None):
        def _l2normalize(v, eps=1e-12):
            return v / (K.sum(v**2) ** 0.5 + eps)

        def power_iteration(W, u):
            _u = u
            _v = _l2normalize(K.dot(_u, K.transpose(W)))
            _u = _l2normalize(K.dot(_v, W))
            return _u, _v

        W_shape = self.kernel.shape.as_list()
        # Flatten the Tensor
        W_reshaped = K.reshape(self.kernel, [-1, W_shape[-1]])
        _u, _v = power_iteration(W_reshaped, self.u)
        # Calculate Sigma
        sigma = K.dot(_v, W_reshaped)
        sigma = K.dot(sigma, K.transpose(_u))
        # normalize it
        W_bar = W_reshaped / sigma
        # reshape weight tensor
        if training in {0, False}:
            W_bar = K.reshape(W_bar, W_shape)
        else:
            with tf.control_dependencies([self.u.assign(_u)]):
                W_bar = K.reshape(W_bar, W_shape)
        output = K.dot(inputs, W_bar)
        if self.use_bias:
            output = K.bias_add(output, self.bias, data_format="channels_last")
        if self.activation is not None:
            output = self.activation(output)
        return output


class _ConvSN(Layer):
    def __init__(
        self,
        rank,
        filters,
        kernel_size,
        strides=1,
        padding="valid",
        data_format=None,
        dilation_rate=1,
        activation=None,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        spectral_normalization=True,
        **kwargs
    ):
        super(_ConvSN, self).__init__(**kwargs)
        self.rank = rank
        self.filters = filters
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, rank,
                                                      "kernel_size")
        self.strides = conv_utils.normalize_tuple(strides, rank, "strides")
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.dilation_rate = conv_utils.normalize_tuple(
            dilation_rate, rank, "dilation_rate"
        )
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(ndim=self.rank + 2)
        self.spectral_normalization = spectral_normalization
        self.u = None

    def _l2normalize(self, v, eps=1e-12):
        return v / (K.sum(v**2) ** 0.5 + eps)

    def power_iteration(self, u, W):
        """
        Accroding the paper, we only need to do power iteration one time.
        """
        v = self._l2normalize(K.dot(u, K.transpose(W)))
        u = self._l2normalize(K.dot(v, W))
        return u, v

    def build(self, input_shape):
        if self.data_format == "channels_first":
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError(
                "The channel dimension of the inputs "
                "should be defined. Found `None`."
            )
        input_dim = input_shape[channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.filters)

        self.kernel = self.add_weight(
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            name="kernel",
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )

        # Spectral Normalization
        if self.spectral_normalization:
            self.u = self.add_weight(
                shape=tuple([1, self.kernel.shape.as_list()[-1]]),
                initializer=initializers.RandomNormal(0, 1),
                name="sn",
                trainable=False,
            )

        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.filters,),
                initializer=self.bias_initializer,
                name="bias",
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
            )
        else:
            self.bias = None
        # Set input spec.
        self.input_spec = InputSpec(ndim=self.rank + 2,
                                    axes={channel_axis: input_dim})
        self.built = True

    def call(self, inputs):
        def _l2normalize(v, eps=1e-12):
            return v / (K.sum(v**2) ** 0.5 + eps)

        def power_iteration(W, u):
            _u = u
            _v = _l2normalize(K.dot(_u, K.transpose(W)))
            _u = _l2normalize(K.dot(_v, W))
            return _u, _v

        if self.spectral_normalization:
            W_shape = self.kernel.shape.as_list()
            # Flatten the Tensor
            W_reshaped = K.reshape(self.kernel, [-1, W_shape[-1]])
            _u, _v = power_iteration(W_reshaped, self.u)
            # Calculate Sigma
            sigma = K.dot(_v, W_reshaped)
            sigma = K.dot(sigma, K.transpose(_u))
            # normalize it
            W_bar = W_reshaped / sigma
            # reshape weight tensor
            if training in {0, False}:
                W_bar = K.reshape(W_bar, W_shape)
            else:
                with tf.control_dependencies([self.u.assign(_u)]):
                    W_bar = K.reshape(W_bar, W_shape)

            # update weitht
            self.kernel = W_bar

        if self.rank == 1:
            outputs = K.conv1d(
                inputs,
                self.kernel,
                strides=self.strides[0],
                padding=self.padding,
                data_format=self.data_format,
                dilation_rate=self.dilation_rate[0],
            )
        if self.rank == 2:
            outputs = K.conv2d(
                inputs,
                self.kernel,
                strides=self.strides,
                padding=self.padding,
                data_format=self.data_format,
                dilation_rate=self.dilation_rate,
            )
        if self.rank == 3:
            outputs = K.conv3d(
                inputs,
                self.kernel,
                strides=self.strides,
                padding=self.padding,
                data_format=self.data_format,
                dilation_rate=self.dilation_rate,
            )

        if self.use_bias:
            outputs = K.bias_add(outputs, self.bias,
                                 data_format=self.data_format)

        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def compute_output_shape(self, input_shape):
        if self.data_format == "channels_last":
            space = input_shape[1:-1]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i],
                )
                new_space.append(new_dim)
            return (input_shape[0],) + tuple(new_space) + (self.filters,)
        if self.data_format == "channels_first":
            space = input_shape[2:]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i],
                )
                new_space.append(new_dim)
            return (input_shape[0], self.filters) + tuple(new_space)

    def get_config(self):
        config = {
            "rank": self.rank,
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "padding": self.padding,
            "data_format": self.data_format,
            "dilation_rate": self.dilation_rate,
            "activation": activations.serialize(self.activation),
            "use_bias": self.use_bias,
            "kernel_initializer": initializers.serialize(
                self.kernel_initializer),
            "bias_initializer": initializers.serialize(
                self.bias_initializer),
            "kernel_regularizer": regularizers.serialize(
                self.kernel_regularizer),
            "bias_regularizer": regularizers.serialize(
                self.bias_regularizer),
            "activity_regularizer": regularizers.serialize(
                self.activity_regularizer),
            "kernel_constraint": constraints.serialize(
                self.kernel_constraint),
            "bias_constraint": constraints.serialize(
                self.bias_constraint),
        }
        base_config = super(_Conv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ConvSN2D(Conv2D):
    def build(self, input_shape):
        if self.data_format == "channels_first":
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError(
                "The channel dimension of the inputs "
                "should be defined. Found `None`."
            )
        input_dim = input_shape[channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.filters)

        self.kernel = self.add_weight(
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            name="kernel",
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )

        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.filters,),
                initializer=self.bias_initializer,
                name="bias",
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
            )
        else:
            self.bias = None

        self.u = self.add_weight(
            shape=tuple([1, self.kernel.shape.as_list()[-1]]),
            initializer=initializers.RandomNormal(0, 1),
            name="sn",
            trainable=False,
        )

        # Set input spec.
        self.input_spec = InputSpec(ndim=self.rank + 2,
                                    axes={channel_axis: input_dim})
        self.built = True

    def call(self, inputs, training=None):
        def _l2normalize(v, eps=1e-12):
            return v / (K.sum(v**2) ** 0.5 + eps)

        def power_iteration(W, u):
            # Accroding the paper, we only need to do power iteration one time.
            _u = u
            _v = _l2normalize(K.dot(_u, K.transpose(W)))
            _u = _l2normalize(K.dot(_v, W))
            return _u, _v

        # Spectral Normalization
        W_shape = self.kernel.shape.as_list()
        # Flatten the Tensor
        W_reshaped = K.reshape(self.kernel, [-1, W_shape[-1]])
        _u, _v = power_iteration(W_reshaped, self.u)
        # Calculate Sigma
        sigma = K.dot(_v, W_reshaped)
        sigma = K.dot(sigma, K.transpose(_u))
        # normalize it
        W_bar = W_reshaped / sigma
        # reshape weight tensor
        if training in {0, False}:
            W_bar = K.reshape(W_bar, W_shape)
        else:
            with tf.control_dependencies([self.u.assign(_u)]):
                W_bar = K.reshape(W_bar, W_shape)

        outputs = K.conv2d(
            inputs,
            W_bar,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate,
        )
        if self.use_bias:
            outputs = K.bias_add(outputs, self.bias,
                                 data_format=self.data_format)
        if self.activation is not None:
            return self.activation(outputs)
        return outputs


class ConvSN1D(Conv1D):
    def build(self, input_shape):
        if self.data_format == "channels_first":
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError(
                "The channel dimension of the inputs "
                "should be defined. Found `None`."
            )
        input_dim = input_shape[channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.filters)

        self.kernel = self.add_weight(
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            name="kernel",
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )

        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.filters,),
                initializer=self.bias_initializer,
                name="bias",
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
            )
        else:
            self.bias = None

        self.u = self.add_weight(
            shape=tuple([1, self.kernel.shape.as_list()[-1]]),
            initializer=initializers.RandomNormal(0, 1),
            name="sn",
            trainable=False,
        )
        # Set input spec.
        self.input_spec = InputSpec(ndim=self.rank + 2,
                                    axes={channel_axis: input_dim})
        self.built = True

    def call(self, inputs, training=None):
        def _l2normalize(v, eps=1e-12):
            return v / (K.sum(v**2) ** 0.5 + eps)

        def power_iteration(W, u):
            # Accroding the paper, we only need to do power iteration one time.
            _u = u
            _v = _l2normalize(K.dot(_u, K.transpose(W)))
            _u = _l2normalize(K.dot(_v, W))
            return _u, _v

        # Spectral Normalization
        W_shape = self.kernel.shape.as_list()
        # Flatten the Tensor
        W_reshaped = K.reshape(self.kernel, [-1, W_shape[-1]])
        _u, _v = power_iteration(W_reshaped, self.u)
        # Calculate Sigma
        sigma = K.dot(_v, W_reshaped)
        sigma = K.dot(sigma, K.transpose(_u))
        # normalize it
        W_bar = W_reshaped / sigma
        # reshape weight tensor
        if training in {0, False}:
            W_bar = K.reshape(W_bar, W_shape)
        else:
            with tf.control_dependencies([self.u.assign(_u)]):
                W_bar = K.reshape(W_bar, W_shape)

        outputs = K.conv1d(
            inputs,
            W_bar,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate,
        )
        if self.use_bias:
            outputs = K.bias_add(outputs, self.bias,
                                 data_format=self.data_format)
        if self.activation is not None:
            return self.activation(outputs)
        return outputs


class ConvSN3D(Conv3D):
    def build(self, input_shape):
        if self.data_format == "channels_first":
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError(
                "The channel dimension of the inputs "
                "should be defined. Found `None`."
            )
        input_dim = input_shape[channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.filters)

        self.kernel = self.add_weight(
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            name="kernel",
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )

        self.u = self.add_weight(
            shape=tuple([1, self.kernel.shape.as_list()[-1]]),
            initializer=initializers.RandomNormal(0, 1),
            name="sn",
            trainable=False,
        )

        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.filters,),
                initializer=self.bias_initializer,
                name="bias",
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
            )
        else:
            self.bias = None
        # Set input spec.
        self.input_spec = InputSpec(ndim=self.rank + 2,
                                    axes={channel_axis: input_dim})
        self.built = True

    def call(self, inputs, training=None):
        def _l2normalize(v, eps=1e-12):
            return v / (K.sum(v**2) ** 0.5 + eps)

        def power_iteration(W, u):
            # Accroding the paper, we only need to do power iteration one time.
            _u = u
            _v = _l2normalize(K.dot(_u, K.transpose(W)))
            _u = _l2normalize(K.dot(_v, W))
            return _u, _v

        # Spectral Normalization
        W_shape = self.kernel.shape.as_list()
        # Flatten the Tensor
        W_reshaped = K.reshape(self.kernel, [-1, W_shape[-1]])
        _u, _v = power_iteration(W_reshaped, self.u)
        # Calculate Sigma
        sigma = K.dot(_v, W_reshaped)
        sigma = K.dot(sigma, K.transpose(_u))
        # normalize it
        W_bar = W_reshaped / sigma
        # reshape weight tensor
        if training in {0, False}:
            W_bar = K.reshape(W_bar, W_shape)
        else:
            with tf.control_dependencies([self.u.assign(_u)]):
                W_bar = K.reshape(W_bar, W_shape)

        outputs = K.conv3d(
            inputs,
            W_bar,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate,
        )
        if self.use_bias:
            outputs = K.bias_add(outputs, self.bias,
                                 data_format=self.data_format)
        if self.activation is not None:
            return self.activation(outputs)
        return outputs


class EmbeddingSN(Embedding):
    def build(self, input_shape):
        self.embeddings = self.add_weight(
            shape=(self.input_dim, self.output_dim),
            initializer=self.embeddings_initializer,
            name="embeddings",
            regularizer=self.embeddings_regularizer,
            constraint=self.embeddings_constraint,
            dtype=self.dtype,
        )

        self.u = self.add_weight(
            shape=tuple([1, self.embeddings.shape.as_list()[-1]]),
            initializer=initializers.RandomNormal(0, 1),
            name="sn",
            trainable=False,
        )

        self.built = True

    def call(self, inputs):
        if K.dtype(inputs) != "int32":
            inputs = K.cast(inputs, "int32")

        def _l2normalize(v, eps=1e-12):
            return v / (K.sum(v**2) ** 0.5 + eps)

        def power_iteration(W, u):
            # Accroding the paper, we only need to do power iteration one time.
            _u = u
            _v = _l2normalize(K.dot(_u, K.transpose(W)))
            _u = _l2normalize(K.dot(_v, W))
            return _u, _v

        W_shape = self.embeddings.shape.as_list()
        # Flatten the Tensor
        W_reshaped = K.reshape(self.embeddings, [-1, W_shape[-1]])
        _u, _v = power_iteration(W_reshaped, self.u)
        # Calculate Sigma
        sigma = K.dot(_v, W_reshaped)
        sigma = K.dot(sigma, K.transpose(_u))
        # normalize it
        W_bar = W_reshaped / sigma
        # reshape weight tensor
        if training in {0, False}:
            W_bar = K.reshape(W_bar, W_shape)
        else:
            with tf.control_dependencies([self.u.assign(_u)]):
                W_bar = K.reshape(W_bar, W_shape)
        self.embeddings = W_bar

        out = K.gather(self.embeddings, inputs)
        return out


class ConvSN2DTranspose(Conv2DTranspose):
    def build(self, input_shape):
        if len(input_shape) != 4:
            raise ValueError(
                "Inputs should have rank 4; Received input shape:",
                str(input_shape),
            )
        if self.data_format == "channels_first":
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError(
                "The channel dimension of the inputs "
                "should be defined. Found `None`."
            )
        input_dim = input_shape[channel_axis]
        kernel_shape = self.kernel_size + (self.filters, input_dim)

        self.kernel = self.add_weight(
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            name="kernel",
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.filters,),
                initializer=self.bias_initializer,
                name="bias",
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
            )
        else:
            self.bias = None

        self.u = self.add_weight(
            shape=tuple([1, self.kernel.shape.as_list()[-1]]),
            initializer=initializers.RandomNormal(0, 1),
            name="sn",
            trainable=False,
        )

        # Set input spec.
        self.input_spec = InputSpec(ndim=4, axes={channel_axis: input_dim})
        self.built = True

    def call(self, inputs):
        input_shape = K.shape(inputs)
        batch_size = input_shape[0]
        if self.data_format == "channels_first":
            h_axis, w_axis = 2, 3
        else:
            h_axis, w_axis = 1, 2

        height, width = input_shape[h_axis], input_shape[w_axis]
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.strides
        if self.output_padding is None:
            out_pad_h = out_pad_w = None
        else:
            out_pad_h, out_pad_w = self.output_padding

        # Infer the dynamic output shape:
        out_height = conv_utils.deconv_length(
            height, stride_h, kernel_h, self.padding, out_pad_h
        )
        out_width = conv_utils.deconv_length(
            width, stride_w, kernel_w, self.padding, out_pad_w
        )
        if self.data_format == "channels_first":
            output_shape = (batch_size, self.filters, out_height, out_width)
        else:
            output_shape = (batch_size, out_height, out_width, self.filters)

        # Spectral Normalization
        def _l2normalize(v, eps=1e-12):
            return v / (K.sum(v**2) ** 0.5 + eps)

        def power_iteration(W, u):
            # Accroding the paper, we only need to do power iteration one time.
            _u = u
            _v = _l2normalize(K.dot(_u, K.transpose(W)))
            _u = _l2normalize(K.dot(_v, W))
            return _u, _v

        W_shape = self.kernel.shape.as_list()
        # Flatten the Tensor
        W_reshaped = K.reshape(self.kernel, [-1, W_shape[-1]])
        _u, _v = power_iteration(W_reshaped, self.u)
        # Calculate Sigma
        sigma = K.dot(_v, W_reshaped)
        sigma = K.dot(sigma, K.transpose(_u))
        # normalize it
        W_bar = W_reshaped / sigma
        # reshape weight tensor
        if training in {0, False}:
            W_bar = K.reshape(W_bar, W_shape)
        else:
            with tf.control_dependencies([self.u.assign(_u)]):
                W_bar = K.reshape(W_bar, W_shape)
        self.kernel = W_bar

        outputs = K.conv2d_transpose(
            inputs,
            self.kernel,
            output_shape,
            self.strides,
            padding=self.padding,
            data_format=self.data_format,
        )

        if self.use_bias:
            outputs = K.bias_add(outputs, self.bias,
                                 data_format=self.data_format)

        if self.activation is not None:
            return self.activation(outputs)
        return outputs


class GlobalSumPooling2D(_GlobalPooling2D):
    """Global sum pooling operation for spatial data.
    # Arguments
        data_format: A string,
            one of `channels_last` (default) or `channels_first`.
            The ordering of the dimensions in the inputs.
            `channels_last` corresponds to inputs with shape
            `(batch, height, width, channels)` while `channels_first`
            corresponds to inputs with shape
            `(batch, channels, height, width)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".
    # Input shape
        - If `data_format='channels_last'`:
            4D tensor with shape:
            `(batch_size, rows, cols, channels)`
        - If `data_format='channels_first'`:
            4D tensor with shape:
            `(batch_size, channels, rows, cols)`
    # Output shape
        2D tensor with shape:
        `(batch_size, channels)`
    """

    def call(self, inputs):
        if self.data_format == "channels_last":
            return K.sum(inputs, axis=[1, 2])
        else:
            return K.sum(inputs, axis=[2, 3])


def matMul1(tensors, gamma=0.0):
    input_shape = K.int_shape(tensors[2])
    a = K.reshape(
        tensors[0],
        (None, input_shape[0 + 1] * input_shape[1 + 1],
         input_shape[2 + 1] // 8),
    )

    b = K.reshape(
        tensors[1],
        (None, input_shape[0 + 1] * input_shape[1 + 1],
         input_shape[2 + 1] // 8),
    )
    b = K.permute_dimensions(b, (1, 0))
    s = K.batch_dot(a, b)  # (f_flat.shape[-1], f_flat.shape[0])) ) [bs, N, N]

    beta = K.softmax(s)
    o = K.batch_dot(
        beta,
        K.reshape(
            tensors[2],
            (None, input_shape[0 + 1] * input_shape[1 + 1],
             input_shape[2 + 1]),
        ),
    )  # [bs, N, C]

    o = gamma * K.reshape(
        o, (None, input_shape[0 + 1], input_shape[1 + 1], 256)
    )  # [bs, h, w, C]

    return o


def outshape(input_shape):
    return list(input_shape[2])


def _conv_layer(filters, kernel_size, strides=(1, 1),
                padding="same", name=None):
    return Conv2D(
        filters,
        kernel_size,
        strides=strides,
        padding=padding,
        use_bias=True,
        kernel_initializer="he_normal",
        name=name,
    )


def _normalize_depth_vars(depth_k, depth_v, filters):
    """
    Accepts depth_k and depth_v as either floats or integers
    and normalizes them to integers.
    Args:
        depth_k: float or int.
        depth_v: float or int.
        filters: number of output filters.
    Returns:
        depth_k, depth_v as integers.
    """

    if type(depth_k) == float:
        depth_k = int(filters * depth_k)
    else:
        depth_k = int(depth_k)

    if type(depth_v) == float:
        depth_v = int(filters * depth_v)
    else:
        depth_v = int(depth_v)

    return depth_k, depth_v


def ResBlock(
    input_shape,
    sampling=None,
    trainable_sortcut=True,
    spectral_normalization=False,
    batch_normalization=True,
    bn_momentum=0.9,
    bn_epsilon=0.00002,
    channels=256,
    k_size=3,
    summary=False,
    plot=False,
    name=None,
):
    """
    ResBlock(input_shape, sampling=None, trainable_sortcut=True,
             spectral_normalization=False, batch_normalization=True,
             bn_momentum=0.9, bn_epsilon=0.00002,
             channels=256, k_size=3, summary=False,
             plot=False, plot_name='res_block.png')""

    Build ResBlock as keras Model
    sampleing = 'up' for upsampling
                'down' for downsampling(AveragePooling)
                None for none

    """
    # input_shape = input_layer.sahpe.as_list()

    res_block_input = Input(shape=input_shape)

    if batch_normalization:
        res_block_1 = BatchNormalization(momentum=bn_momentum,
                                         epsilon=bn_epsilon)(
            res_block_input
        )
    else:
        res_block_1 = res_block_input

    res_block_1 = Activation("relu")(res_block_1)

    if sampling == "up":
        res_block_1 = UpSampling2D()(res_block_1)
    else:
        pass

    if spectral_normalization:
        res_block_1 = cc.WrapPadding2D(padding=(0, 1))(res_block_1)
        res_block_1 = cc.NearestPadding2D(padding=(1, 0))(res_block_1)
        res_block_1 = cc.ConvSN2D(
            channels,
            k_size,
            strides=1,
            padding="valid",
            kernel_initializer="glorot_uniform",
        )(res_block_1)

    else:
        res_block_1 = cc.WrapPadding2D(padding=(0, 1))(res_block_1)
        res_block_1 = cc.NearestPadding2D(padding=(1, 0))(res_block_1)
        res_block_1 = Conv2D(
            channels,
            k_size,
            strides=1,
            padding="valid",
            kernel_initializer="glorot_uniform",
        )(res_block_1)

    if batch_normalization:
        res_block_2 = BatchNormalization(momentum=bn_momentum,
                                         epsilon=bn_epsilon)(
            res_block_1
        )
    else:
        res_block_2 = res_block_1
    res_block_2 = Activation("relu")(res_block_2)

    if spectral_normalization:
        res_block_2 = WrapPadding2D(padding=(0, 1))(res_block_2)
        res_block_2 = NearestPadding2D(padding=(1, 0))(res_block_2)
        res_block_2 = ConvSN2D(
            channels,
            k_size,
            strides=1,
            padding="valid",
            kernel_initializer="glorot_uniform",
        )(res_block_2)
    else:
        res_block_2 = WrapPadding2D(padding=(0, 1))(res_block_2)
        res_block_2 = NearestPadding2D(padding=(1, 0))(res_block_2)
        res_block_2 = Conv2D(
            channels,
            k_size,
            strides=1,
            padding="valid",
            kernel_initializer="glorot_uniform",
        )(res_block_2)

    if sampling == "down":
        res_block_2 = AveragePooling2D()(res_block_2)
    else:
        pass

    if sampling == "up":
        short_cut = UpSampling2D()(res_block_input)
    elif sampling == "down":
        short_cut = AveragePooling2D()(res_block_input)
    else:
        short_cut = res_block_input

    if trainable_sortcut:
        if spectral_normalization:
            short_cut = cc.ConvSN2D(
                channels,
                1,
                strides=1,
                padding="valid",
                kernel_initializer="glorot_uniform",
            )(short_cut)
        else:
            short_cut = Conv2D(
                channels,
                1,
                strides=1,
                padding="valid",
                kernel_initializer="glorot_uniform",
            )(short_cut)
    else:
        short_cut = res_block_input
