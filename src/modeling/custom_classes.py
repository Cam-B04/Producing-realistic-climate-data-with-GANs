

import tensorflow as tf
from keras.utils import conv_utils
#from tensorflow.contrib import keras
from keras.layers import InputSpec 
from keras.legacy import interfaces
from keras.utils.generic_utils import transpose_shape
#from tensorflow.python.keras import *

#import keras.layers.merge
from keras.optimizers import Adam, RMSprop, SGD
from keras.layers import Input, Add, Lambda, Layer, Dense, Reshape, BatchNormalization, Input, Dropout, Dense, Lambda, Conv2D, MaxPooling2D, UpSampling2D
from keras.layers import Flatten, Conv2DTranspose, Activation, Cropping2D, ZeroPadding2D, Conv2DTranspose, AveragePooling2D
from keras.models import Model
import keras.backend as K
import os

from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from functools import partial
from keras import metrics, regularizers
from keras.losses import mse, binary_crossentropy, kullback_leibler_divergence,mae
from keras.initializers import glorot_uniform

def periodic_padding(image, padding):
    '''
    Create a periodic padding (wrap) around the image, to emulate periodic boundary conditions
    '''
    #print(padding)
    #pad_h = tf.cast(padding[0][0], tf.int32)
    #pad_w = tf.cast(padding[1][1], tf.int32)
    assert len(padding) == 2
    assert len(padding[0]) == 2
    assert len(padding[1]) == 2
    #data_format = normalize_data_format(data_format)

    pad_h = padding[0][0]
    pad_w = padding[1][1]
    padding = (pad_h, pad_w)
    if padding[0] != 0 :
        upper_pad = image[:,-1*padding[0]:,:]
        lower_pad = image[:,:padding[0],:]
        padded_image = tf.concat([upper_pad, image, lower_pad], axis=1)
    
    elif padding[1] != 0 :
        left_pad = image[:,:,-1*padding[1]:]
        right_pad = image[:,:,:padding[1]]
        padded_image = tf.concat([left_pad, image, right_pad], axis=2)
    
    return padded_image

def periodic_paddingv2(image, padding=1):
    '''
    Create a periodic padding (wrap) around the image, to emulate periodic boundary conditions
    '''

    rows, columns = image.shape
    
    # create left matrix
    left_corner_diagonal = tf.eye(padding)
    left_filled_zeros = tf.zeros([padding,rows.value-padding])
    
    left_upper = tf.concat([left_filled_zeros, left_corner_diagonal], axis=1)
    left_center_diagonal = tf.eye(rows.value)
    left_lower = tf.concat([left_corner_diagonal,left_filled_zeros], axis=1)
    
    left_matrix = tf.concat([left_upper, left_center_diagonal, left_lower], axis=0)
    
    # create right matrix
    right_corner_diagonal = tf.eye(padding)
    right_filled_zeros = tf.zeros([columns.value-padding,padding])
    
    right_left_side = tf.concat([right_filled_zeros, right_corner_diagonal], axis=0)
    right_center_diagonal = tf.eye(columns.value)
    right_right_side = tf.concat([right_corner_diagonal,right_filled_zeros], axis=0)
    
    right_matrix = tf.concat([right_left_side, right_center_diagonal, right_right_side], axis=1)
    
    # left and right matrices are immutable
    padded_image = tf.matmul(left_matrix, tf.matmul(image, right_matrix))

    return padded_image

def nearest_padding(image, padding):
    '''
    Create a periodic padding (wrap) around the image, to emulate periodic boundary conditions
    '''
    pad_h = padding[0][0]
    pad_w = padding[1][1]
    padding = (pad_h, pad_w)

    if padding[0] != 0 :
        upper_pad = image[:,:1,:]
        lower_pad = image[:,-1:,:]
        L = [upper_pad, image, lower_pad]
        for i in range(padding[0]-1):
            L.insert(0, upper_pad)
            L.insert(-1, lower_pad)
    
        padded_image = tf.concat(L, axis=1)
    
    elif padding[1] != 0 :

        left_pad = image[:,:,:1]
        right_pad = image[:,:,-1:]
        for i in range(padding[0]-1):
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
    def __init__(self, padding, data_format='channels_last', **kwargs):
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
        padding_all_dims = transpose_shape(padding_all_dims,
                                           self.data_format,
                                           spatial_axes)
        output_shape = list(input_shape)
        for dim in range(len(output_shape)):
            if output_shape[dim] is not None:
                output_shape[dim] += sum(padding_all_dims[dim])
        return tuple(output_shape)

    def get_config(self):
        config = {'padding': self.padding,
                  'data_format': self.data_format}
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

    #@interfaces.legacy_zeropadding2d_support
    def __init__(self, padding=(1, 1), data_format='channels_last', **kwargs):
        #self.padding = padding
        if isinstance(padding, int):
            normalized_padding = ((padding, padding), (padding, padding))
        elif hasattr(padding, '__len__'):
            if len(padding) != 2:
                raise ValueError('`padding` should have two elements. '
                                 'Found: ' + str(padding))
            height_padding = conv_utils.normalize_tuple(padding[0], 2,
                                                        '1st entry of padding')
            width_padding = conv_utils.normalize_tuple(padding[1], 2,
                                                       '2nd entry of padding')
            normalized_padding = (height_padding, width_padding)
        else:
            raise ValueError('`padding` should be either an int, '
                             'a tuple of 2 ints '
                             '(symmetric_height_pad, symmetric_width_pad), '
                             'or a tuple of 2 tuples of 2 ints '
                             '((top_pad, bottom_pad), (left_pad, right_pad)). '
                             'Found: ' + str(padding))
        super(WrapPadding2D, self).__init__(normalized_padding,
                                            data_format,
                                            **kwargs)

    def call(self, inputs):
        return periodic_padding(inputs, padding = self.padding)
    
    #@property
    #def outbound_nodes(self):
    #    if hasattr(self, '_outbound_nodes'):
    #        print("outbound_nodes called but _outbound_nodes found")
    #    return getattr(self, '_outbound_nodes', [])                     

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
        elif hasattr(padding, '__len__'):
            if len(padding) != 2:
                raise ValueError('`padding` should have two elements. '
                                 'Found: ' + str(padding))
            height_padding = conv_utils.normalize_tuple(padding[0], 2,
                                                        '1st entry of padding')
            width_padding = conv_utils.normalize_tuple(padding[1], 2,
                                                       '2nd entry of padding')
            normalized_padding = (height_padding, width_padding)
        else:
            raise ValueError('`padding` should be either an int, '
                             'a tuple of 2 ints '
                             '(symmetric_height_pad, symmetric_width_pad), '
                             'or a tuple of 2 tuples of 2 ints '
                             '((top_pad, bottom_pad), (left_pad, right_pad)). '
                             'Found: ' + str(padding))
        super(NearestPadding2D, self).__init__(normalized_padding,
                                            data_format,
                                            **kwargs)

    def call(self, inputs):
        return nearest_padding(inputs, padding = self.padding)
    
    #@property
    #def outbound_nodes(self):
    #    if hasattr(self, '_outbound_nodes'):
    #        print("outbound_nodes called but _outbound_nodes found")
    #    return getattr(self, '_outbound_nodes', []) 

class PB_layer_pad(Layer):

    def __init__(self,img_shape, axis, padding, **kwargs):
        #self.output_dim = output_dim
        self.img_rows = img_shape[0]
        self.img_cols = img_shape[1]
        self.channels = img_shape[2]
        self.padding = padding
        self.axis = axis
        super(PB_layer_pad, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(1,1),
                                      initializer='uniform',
                                      trainable=False)
        super(PB_layer_pad, self).build(input_shape)  # Be sure to call this at the end

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
        

        if isinstance(self.axis,int):
            self.axis_ = (self.axis,)
        if isinstance(self.padding,int):
            self.padding_ = (self.padding,)
    
        ndim = 4
        for ax,p in zip(self.axis_,self.padding_):
            # create a slice object that selects everything from all axes,
            # except only 0:p for the specified for right, and -p: for left
    
            ind_right = [slice(-p,None) if i == ax else slice(None) for i in range(ndim)]
            ind_left = [slice(0, p) if i == ax else slice(None) for i in range(ndim)]
            right = tensor[ind_right]
            print(ind_right)
            left = tensor[ind_left]
            middle = tensor
            tensor = tf.concat([right,middle,left], axis=ax)
    
        return tensor

    def compute_output_shape(self, input_shape):
        if self.axis_[0] == 1:
            #shape = (None,64,132,5)
            shape = (input_shape[0], input_shape[1]+2*self.padding, input_shape[2], input_shape[3])
        elif self.axis_[0] == 2:
            #shape = (None,64,132,5)
            shape = (input_shape[0], input_shape[1], input_shape[2]+2*self.padding, input_shape[3])
            
        #else:
            #shape = (input_shape[0], input_shape[1], input_shape[2]+self.padding, input_shape[3])
        return shape

def identity_block(X, f, filters, stage, block):
    
    # Defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value
    X_shortcut = X
    
    # First component of main path
    X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
    #X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    
    # Second component of main path
    X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1, 1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)
    #X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path 
    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1, 1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0))(X)
    #X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    
    return X

def convolutional_block(X, f, filters, stage, block, s=2):

    # Defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value
    X_shortcut = X

    ##### MAIN PATH #####
    # First component of main path 
    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '2a', kernel_initializer=glorot_uniform(seed=0))(X)
    #X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # Second component of main path
    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b', kernel_initializer=glorot_uniform(seed=0))(X)
    #X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path
    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c', kernel_initializer=glorot_uniform(seed=0))(X)
    #X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    ##### SHORTCUT PATH #### 
    X_shortcut = Conv2D(filters=F3, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '1', kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    #X_shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X


def ResNet50(input_shape = (64, 64, 3), classes = 6, warp = True):
    from . import WrapPadding2D 
    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)

    
    # WarpPadding
    if warp:
    	X = WarpPadding2D(padding = (0, 6))(X_input)
    else:
    	X = Conv2D(81, (7, 7), strides = (1, 1), name = 'conv1', kernel_initializer = glorot_uniform(seed=0))(X_input)
    # Stage 1
    X = Conv2D(81, (7, 7), strides = (2, 2), name = 'conv1', kernel_initializer = glorot_uniform(seed=0))(X)
    #X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 2
    X = convolutional_block(X, f = 3, filters = [64, 64, 128], stage = 2, block='a', s = 2)
    X = identity_block(X, 3, [81, 81, 128], stage=2, block='b')
    X = identity_block(X, 3, [81, 81, 128], stage=2, block='c')

    # Stage 3
    X = convolutional_block(X, f=3, filters=[128, 128, 128], stage=3, block='a', s=2)
    X = identity_block(X, 3, [128, 128, 256], stage=3, block='b')
    X = identity_block(X, 3, [128, 128, 256], stage=3, block='c')
    X = identity_block(X, 3, [128, 128, 256], stage=3, block='d')

    # Stage 4
    X = convolutional_block(X, f=3, filters=[128, 128, 256], stage=4, block='a', s=2)
    X = identity_block(X, 3, [256, 256, 256], stage=4, block='b')
    X = identity_block(X, 3, [256, 256, 256], stage=4, block='c')
    X = identity_block(X, 3, [256, 256, 256], stage=4, block='d')
    X = identity_block(X, 3, [256, 256, 256], stage=4, block='e')
    #X = identity_block(X, 3, [128, 128, 256], stage=4, block='f')

    # Stage 5
    #X = convolutional_block(X, f=3, filters=[512, 512, 2048], stage=5, block='a', s=2)
    #X = identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
    #X = identity_block(X, 3, [512, 512, 2048], stage=5, block='c')

    # AVGPOOL
    X = AveragePooling2D(pool_size=(2,2), padding='same')(X)

    # Output layer
    X = Flatten()(X)
    X = Dense(1, kernel_initializer = glorot_uniform(seed=0))(X)
    
    
    # Create model
    return Model(inputs = X_input, outputs = X, name='ResNet50')

def inv_identity_block(X, f, filters, stage, block):
    
    # Defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value
    X_shortcut = X
    
    # First component of main path
    X = Conv2DTranspose(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
    #X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    
    # Second component of main path
    X = Conv2DTranspose(filters = F2, kernel_size = (f, f), strides = (1, 1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)
    #X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path 
    X = Conv2DTranspose(filters = F3, kernel_size = (1, 1), strides = (1, 1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0))(X)
    #X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    
    return X

def inv_convolutional_block(X, f, filters, stage, block, s=2):

    # Defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value
    X_shortcut = X

    ##### MAIN PATH #####
    # First component of main path 
    X = Conv2DTranspose(filters=F1, kernel_size=(1, 1), strides=(s, s), output_padding = 1, padding='valid', name=conv_name_base + '2a', kernel_initializer=glorot_uniform(seed=0))(X)
    #X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # Second component of main path
    X = Conv2DTranspose(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b', kernel_initializer=glorot_uniform(seed=0))(X)
    #X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path
    X = Conv2DTranspose(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c', kernel_initializer=glorot_uniform(seed=0))(X)
    #X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    ##### SHORTCUT PATH #### 
    X_shortcut = Conv2DTranspose(filters=F3, kernel_size=(1, 1), output_padding = 1, strides=(s, s), padding='valid', name=conv_name_base + '1', kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    #X_shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X



import numpy as np
import tensorflow as tf
from keras import backend as K
from keras import Input, Model
from keras.layers import Conv2D, MaxPooling2D, merge, Lambda, Dense, Reshape, regularizers

 
def compact_bilinear(tensors_list):

    def _generate_sketch_matrix(rand_h, rand_s, output_dim):
        """
        Return a sparse matrix used for tensor sketch operation in compact bilinear
        pooling
        Args:
            rand_h: an 1D numpy array containing indices in interval `[0, output_dim)`.
            rand_s: an 1D numpy array of 1 and -1, having the same shape as `rand_h`.
            output_dim: the output dimensions of compact bilinear pooling.
        Returns:
            a sparse matrix of shape [input_dim, output_dim] for tensor sketch.
        """

        # Generate a sparse matrix for tensor count sketch
        rand_h = rand_h.astype(np.int64)
        rand_s = rand_s.astype(np.float32)
        assert (rand_h.ndim == 1 and rand_s.ndim == 1 and len(rand_h) == len(rand_s))
        assert (np.all(rand_h >= 0) and np.all(rand_h < output_dim))

        input_dim = len(rand_h)
        indices = np.concatenate((np.arange(input_dim)[..., np.newaxis],
                                  rand_h[..., np.newaxis]), axis=1)
        sparse_sketch_matrix = tf.sparse_reorder(
            tf.SparseTensor(indices, rand_s, [input_dim, output_dim]))
        return sparse_sketch_matrix

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
    sparse_sketch_matrix1 = _generate_sketch_matrix(rand_h_1, rand_s_1, output_dim)

    # Generate sparse_sketch_matrix2 using rand_h_2 and rand_s_2
    np.random.seed(seed_h_2)
    rand_h_2 = np.random.randint(output_dim, size=input_dim2)
    np.random.seed(seed_s_2)
    rand_s_2 = 2 * np.random.randint(2, size=input_dim2) - 1
    sparse_sketch_matrix2 = _generate_sketch_matrix(rand_h_2, rand_s_2, output_dim)

    # Step 1: Flatten the input tensors and count sketch
    bottom1_flat = tf.reshape(bottom1, [-1, input_dim1])
    bottom2_flat = tf.reshape(bottom2, [-1, input_dim2])

    # Essentially:
    #   sketch1 = bottom1 * sparse_sketch_matrix
    #   sketch2 = bottom2 * sparse_sketch_matrix
    # But tensorflow only supports left multiplying a sparse matrix, so:
    #   sketch1 = (sparse_sketch_matrix.T * bottom1.T).T
    #   sketch2 = (sparse_sketch_matrix.T * bottom2.T).T
    sketch1 = tf.transpose(tf.sparse_tensor_dense_matmul(sparse_sketch_matrix1,
                                                         bottom1_flat, adjoint_a=True, adjoint_b=True))
    sketch2 = tf.transpose(tf.sparse_tensor_dense_matmul(sparse_sketch_matrix2,
                                                         bottom2_flat, adjoint_a=True, adjoint_b=True))

    # Step 2: FFT
    fft1 = tf.fft(tf.complex(real=sketch1, imag=tf.zeros_like(sketch1)))
    fft2 = tf.fft(tf.complex(real=sketch2, imag=tf.zeros_like(sketch2)))

    # Step 3: Elementwise product
    fft_product = tf.multiply(fft1, fft2)

    # Step 4: Inverse FFT and reshape back
    # Compute output shape dynamically: [batch_size, height, width, output_dim]
    cbp_flat = tf.real(tf.ifft(fft_product))

    output_shape = tf.add(tf.multiply(tf.shape(bottom1), [1, 1, 1, 0]),
                          [0, 0, 0, output_dim])
    cbp = tf.reshape(cbp_flat, output_shape)

    # print (cbp.get_shape().as_list())

    return cbp

class cos_acti(Layer):
    """Exponential Linear Unit.
    It follows:
    `f(x) =  alpha * (exp(x) - 1.) for x < 0`,
    `f(x) = x for x >= 0`.
    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
    # Output shape
        Same shape as the input.
    # Arguments
        alpha: scale for the negative factor.
    # References
        - [Fast and Accurate Deep Network Learning by Exponential Linear Units
           (ELUs)](https://arxiv.org/abs/1511.07289v1)
    """

    def __init__(self, **kwargs):
        super(cos_acti, self).__init__(**kwargs)
        #self.supports_masking = True
        #self.alpha = K.cast_to_floatx(alpha)

    def call(self, inputs):
        return tf.math.cos(inputs)

    def get_config(self):
        # config = {'alpha': float(self.alpha)}
        base_config = super(cos_acti, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape























