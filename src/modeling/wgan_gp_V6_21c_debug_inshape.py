import numpy as np

import tensorflow as tf 
##############Memory managment###############


#Setting for memory allocaton of the GPU. 
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = str(0)
set_session(tf.Session(config=config))


import keras.layers.merge
from keras.optimizers import Adam
from keras.layers import Input, ZeroPadding2D, Lambda, Layer, Dense, Reshape,BatchNormalization,Input,Dropout ,Dense, Lambda, Conv2D, MaxPooling2D, UpSampling2D, Flatten,Conv2DTranspose,Activation, Cropping2D
from keras.models import Model
import keras.backend as K
import os

from keras.datasets import mnist
from tensorflow.python.keras.layers.merge import _Merge
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import RMSprop, Adam, SGD
from functools import partial
from keras.layers import Reshape,BatchNormalization,Input,Dropout ,Dense, Lambda, Conv2D, MaxPooling2D, UpSampling2D, Flatten,Conv2DTranspose,Activation, MaxPooling2D
from keras.models import Model
from keras import backend as K
from keras import metrics,regularizers
from keras.losses import mse, binary_crossentropy, kullback_leibler_divergence,mae
from keras.datasets import mnist
import h5py as h5
import copy as cp
import keras
from datetime import datetime as dt
import time
import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame, Series
import sys
from keras.initializers import RandomNormal
import matplotlib.pyplot as plt
from scipy.stats import norm
import os.path
import sys
#from GPyOpt.experiment_design import initial_design
import sklearn as skl
import sys


import datetime 

#from ..legacy import interfaces
import scipy.spatial as ss
import scipy.stats as sst
import scipy.io as sio
from scipy.special import beta,digamma,gamma
from sklearn.neighbors import KernelDensity
from math import log,pi,exp
import numpy.random as nr
import numpy as np
import random
import time
import matplotlib.pyplot as plt
#from cvxopt import matrix,solvers
import pandas as pd

import numpy as np

colortest=['k','0.80','0.60','0.40','0.20']

#np.random.seed(seed=1234)
from scipy.stats import entropy
import scipy as sp
from functools import partial

sys.path.append('./src/modeling')
from custom_classes import *

'''class RandomWeightedAverage(keras.layers.merge._Merge):
    """Provides a (random) weighted average between real and generated image samples"""

    def __init__(self, batch_size=32, **kwargs):
        super().__init__(**kwargs)
        self.bs = batch_size

    def _merge_function(self, inputs):
        alpha = K.random_uniform((self.bs, 1, 1, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])'''

class RandomWeightedAverage(keras.layers.merge._Merge):
    """Provides a (random) weighted average between real and generated image samples"""

    def __init__(self, batch_size=32, **kwargs):
        super().__init__(**kwargs)
        self.bs = batch_size

    def _merge_function(self, inputs):
        #alpha = K.random_uniform((self.bs, 1, 1, 1))
        return inputs[0]

'''class RollLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(RollLayer, self).__init__()

    def build(self, input_shape):
        return
    
    def call(self, input):
        return tf.roll(input, shift = 32, axis = 1)'''

'''def outbound_nodes(self):
    if hasattr(self, '_outbound_nodes'):
        print("outbound_nodes called but _outbound_nodes found")
    return getattr(self, '_outbound_nodes', [])'''

import tensorflow as tf
from keras.utils import conv_utils
#from tensorflow.contrib import keras
#from tensorflow.keras.layers import InputSpec 
from keras.legacy import interfaces
#from keras.utils.generic_utils import transpose_shape


class WGANGP(object):
    """
    Implementation of the Improved Wasserstein GAN as proposed by
    [Gulrajani2017A][] (see also [Arxiv](https://arxiv.org/abs/1704.00028))
    which proposes a new implementation for WGAN initially proposed by [Arjovsky2017A][].

    [Arjovsky2017A]: ./pdf/Arjovsky2017A.pdf "Wasserstein GAN"
    [Gulrajani2017A]: ./pdf/Gulrajani2017A.pdf "Improved Training of Wasserstein GANs"
    """

    def __init__(self, latent_dim=128, target_shape=(64, 128, 22), batch_size=32,
                 optimizerG=None, optimizerC=None, summary=False, n_critic=5,
                 models=None, weights_path = None, first_epoch = 0, gradient_penalty=10, data=None, tfboard = False):

        # If models are given then extract information from them.
        # models == [generator, critic]

        if models is not None:
            self.generator, self.critic = models[0], models[1]
            latent_dim = self.generator.input_shape[1]
            target_shape = self.generator.output_shape[1:]

        self.data = data
        self.tfboard = tfboard
        # Set target property
        self.target_shape = target_shape
        self.first_epoch = first_epoch
        # Design of latent space
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        # Following parameter and optimizer set as recommended in WGAN paper
        self.n_critic = n_critic  # Number of discriminator iteration each epoch.

        if optimizerG is None:
            # optimizer= keras.optimizers.RMSprop(lr=0.00005)
            optimizerG = Adam(lr=0.0001, beta_1=0.5, beta_2=0.999,
                                               epsilon=None, decay=0.0, amsgrad=False)

        if optimizerC is None:
            # optimizerC= keras.optimizers.RMSprop(lr=0.00005)
            optimizerC = Adam(lr=0.0001, beta_1=0.5, beta_2=0.999,
                                               epsilon=None, decay=0.0, amsgrad=False)

        self.optimizerG = optimizerG
        self.optimizerC = optimizerC
        self.gradient_penalty = gradient_penalty  # used for the loss of the critic.

        self.summary = summary

        # Create the computation graphs
        if models is None:
            self.generator = self._build_generator()
            self.critic = self._build_critic()
        #elif models is 'resnet':
            #self.critic = ResNet50(input = (64,128,22) , classes = 1)
            #self.generator = self._build_generator()

        

        if self.summary:
            self.critic.summary()
            self.generator.summary()

        self._set_critic_graph()
        self._set_generator_graph()

    def _build_generator(self):
        import tensorflow as tf
        import keras.backend as K
        #import keras.conv_utils
        intermediate_dim = 256
        #model = Sequential()
        #PB = Lambda(self.PeriodBound)
        tuple_conv=(3,3)
        padd = 'valid'
        #Activation function
        activation_dec='relu'
        #activation_dec=keras.layers.LeakyReLU(alpha=0.2)
        init = keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)
        reg_dec=0.000001

        noise = Input(shape=(self.latent_dim,))
        conv = Dense(8*16*intermediate_dim,kernel_initializer=init,activation=activation_dec,kernel_regularizer=regularizers.l2(reg_dec),
         activity_regularizer=regularizers.l2( reg_dec))(noise)
        #conv =  BatchNormalization(momentum=0.99)(conv)
        conv = Reshape((8,16,intermediate_dim))(conv)
        #conv = WrapPadding2D((0,2))(conv)
        #conv = NearestPadding2D((1,0))(conv)
        
        conv =  Conv2DTranspose(256, (3,3),   padding = padd,kernel_initializer=init,kernel_regularizer=regularizers.l2(reg_dec),#Achanger !!
            activity_regularizer=regularizers.l2( reg_dec))(conv)
        conv =  BatchNormalization(momentum=0.99)(conv)
        conv =  Activation(activation_dec)(conv)
        #conv = WrapPadding2D(padding = (0,2))(conv)
        #conv = NearestPadding2D(padding = (1,0))(conv)
        conv = UpSampling2D(size=(2, 2), data_format=None, interpolation='nearest')(conv)

        conv =  Conv2DTranspose(256, tuple_conv,strides=1, padding= padd,kernel_initializer=init,kernel_regularizer=regularizers.l2(reg_dec),#Achanger !!
           activity_regularizer=regularizers.l2( reg_dec))(conv)
        conv =  BatchNormalization(momentum=0.99)(conv)
        conv =  Activation(activation_dec)(conv)
        #conv = WrapPadding2D(padding = (0,2))(conv)
        #conv = NearestPadding2D((1,0))(conv)
        conv = UpSampling2D(size=(2, 2), data_format=None, interpolation='nearest')(conv)

        conv =  Conv2DTranspose(256, (3,3), strides=1, padding= padd,kernel_initializer=init,kernel_regularizer=regularizers.l2(reg_dec),#Achanger !!
            activity_regularizer=regularizers.l2( reg_dec))(conv)
        conv =  BatchNormalization(momentum=0.99)(conv)
        conv =  Activation(activation_dec)(conv)
        conv = WrapPadding2D(padding = (0,9))(conv)
        #conv = ZeroPadding2D(padding = (0,9))(conv)

        conv = NearestPadding2D((4,0))(conv)
        #conv = ZeroPadding2D((4,0))(conv)
        
        conv =  Conv2DTranspose(256, tuple_conv,strides=1,   padding= padd,  kernel_initializer=init,kernel_regularizer=regularizers.l2(reg_dec),
            activity_regularizer=regularizers.l2( reg_dec))(conv)
        conv = BatchNormalization(momentum=0.99)(conv)
        conv = Activation(activation_dec)(conv)
        conv = WrapPadding2D(padding = (0,4))(conv)
        conv = NearestPadding2D((1,0))(conv)
        #conv = ZeroPadding2D(padding = (0,4))(conv)
        #conv = ZeroPadding2D((1,0))(conv)
        conv =  Conv2DTranspose(256, (3,3),   padding=padd,kernel_initializer=init,kernel_regularizer=regularizers.l2(reg_dec),#Achanger !!
            activity_regularizer=regularizers.l2( reg_dec))(conv)
        conv =  BatchNormalization(momentum=0.99)(conv)
        conv =  Activation(activation_dec)(conv)
        #conv = UpSampling2D(size=(2, 2), data_format=None, interpolation='nearest')(conv)
        conv = WrapPadding2D(padding = (0,4))(conv)
        conv = NearestPadding2D(padding = (1,0))(conv)
        #conv = ZeroPadding2D(padding = (0,4))(conv)
        #conv = ZeroPadding2D((1,0))(conv)
        conv =  Conv2DTranspose(256, (3,3),   padding=padd,kernel_initializer=init,kernel_regularizer=regularizers.l2(reg_dec),#Achanger !!
            activity_regularizer=regularizers.l2( reg_dec))(conv)
        conv =  BatchNormalization(momentum=0.99)(conv)
        conv =  Activation(activation_dec)(conv)
        conv = WrapPadding2D(padding = (0,6))(conv)
        conv = NearestPadding2D(padding = (1,0))(conv)
        #conv = ZeroPadding2D(padding = (0,6))(conv)
        #conv = ZeroPadding2D((1,0))(conv)
        conv =  Conv2DTranspose(256, (3,3),   padding='same',kernel_initializer=init,kernel_regularizer=regularizers.l2(reg_dec),#Achanger !!
            activity_regularizer=regularizers.l2( reg_dec))(conv)


        img =  Conv2D(self.target_shape[-1], (3,3),   padding='valid',kernel_initializer=init,kernel_regularizer=regularizers.l2(reg_dec),#Achanger !!
            activity_regularizer=regularizers.l2( reg_dec))(conv)

        model = Model(inputs = noise,outputs = img)
        #img = model(noise)
        #model.summary()
        return model

    def _build_critic(self):
        DO_rate=0.25 #DropOut rate
        tuple_conv=(3,3) #Size of kernel convolution
        init = keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)

        #activation_enc='relu'
        activation_enc=keras.layers.LeakyReLU(alpha=0.2)
        #activation_enc=keras.layers.ELU(alpha=1.0)


        reg_enc=0.000001

        imgcrit = Input(shape=self.target_shape)

        conv = Conv2D(64, (3,3) ,padding='valid',kernel_initializer=init ,kernel_regularizer=regularizers.l2(reg_enc),
         activity_regularizer=regularizers.l2(reg_enc))(imgcrit)

        conv = activation_enc(conv)

        conv = Conv2D(64, (3,3) ,padding='valid',kernel_initializer=init,kernel_regularizer=regularizers.l2(reg_enc),
         activity_regularizer=regularizers.l2(reg_enc))(conv)
        conv = activation_enc(conv)


        conv = Conv2D(64, (4,4) ,padding='valid',kernel_initializer=init,kernel_regularizer=regularizers.l2(reg_enc),
         activity_regularizer=regularizers.l2(reg_enc))(conv)
        conv = activation_enc(conv)

        conv = Conv2D(128, (3,3),strides=1,padding='valid',kernel_initializer=init,kernel_regularizer=regularizers.l2(reg_enc),
           activity_regularizer=regularizers.l2(reg_enc))(conv)
        conv = activation_enc(conv)

        #conv = WrapPadding2D()(conv)
        #conv = Dropout(DO_rate)(conv)
        conv = Conv2D(128, (4,4) ,padding='valid',kernel_initializer=init,kernel_regularizer=regularizers.l2(reg_enc),
           activity_regularizer=regularizers.l2(reg_enc))(conv)
        #conv14 = BatchNormalization(momentum=0.8)(conv13)
        conv = activation_enc(conv)

        conv = Conv2D(128, tuple_conv,strides=1, padding='valid',kernel_initializer=init,kernel_regularizer=regularizers.l2(reg_enc),
          activity_regularizer=regularizers.l2(reg_enc))(conv)
        #conv17 = BatchNormalization(momentum=0.8)(conv16)
        conv = activation_enc(conv)
        #conv = WrapPadding2D()(conv)
        conv = Conv2D(128, tuple_conv,strides=2, padding='valid',kernel_initializer=init,kernel_regularizer=regularizers.l2(reg_enc),
          activity_regularizer=regularizers.l2(reg_enc))(conv)
        #conv17 = BatchNormalization(momentum=0.8)(conv16)
        conv = activation_enc(conv)

        conv = Conv2D(128, (4,4) ,padding='valid',kernel_initializer=init,kernel_regularizer=regularizers.l2(reg_enc),
           activity_regularizer=regularizers.l2(reg_enc))(conv)
        #conv20 = BatchNormalization(momentum=0.8)(conv19)
        conv = activation_enc(conv)

        conv = Conv2D(128, tuple_conv,strides=1, padding='valid',kernel_initializer=init,kernel_regularizer=regularizers.l2(reg_enc),
          activity_regularizer=regularizers.l2(reg_enc))(conv)
        #conv17 = BatchNormalization(momentum=0.8)(conv16)
        conv = activation_enc(conv)
        #conv = WrapPadding2D()(conv)
        #conv = Dropout(DO_rate)(conv)
        conv = Conv2D(128, (4,4) ,padding='valid',kernel_initializer=init,kernel_regularizer=regularizers.l2(reg_enc),
           activity_regularizer=regularizers.l2(reg_enc))(conv)
        #conv20 = BatchNormalization(momentum=0.8)(conv19)
        conv = activation_enc(conv)

        conv = Conv2D(256, (4,4) ,padding='valid',kernel_initializer=init,kernel_regularizer=regularizers.l2(reg_enc),
          activity_regularizer=regularizers.l2(reg_enc))(conv)
        #conv26 = BatchNormalization(momentum=0.8)(conv25)
        conv = activation_enc(conv)

        conv = Conv2D(256, tuple_conv,strides=1, padding='valid',kernel_initializer=init,kernel_regularizer=regularizers.l2(reg_enc),
         activity_regularizer=regularizers.l2(reg_enc))(conv)
        #conv29 = BatchNormalization(momentum=0.8)(conv28)
        conv = activation_enc(conv)
        
        conv = Conv2D(256, tuple_conv,strides=1, padding='valid',kernel_initializer=init,kernel_regularizer=regularizers.l2(reg_enc),
          activity_regularizer=regularizers.l2(reg_enc))(conv)
        #conv17 = BatchNormalization(momentum=0.8)(conv16)
        conv = activation_enc(conv)
        #conv = WrapPadding2D()(conv)
        #conv = Dropout(DO_rate)(conv)
        conv = Conv2D(256, (4,4) ,padding='valid',kernel_initializer=init,kernel_regularizer=regularizers.l2(reg_enc),
           activity_regularizer=regularizers.l2(reg_enc))(conv)
        #conv20 = BatchNormalization(momentum=0.8)(conv19)
        conv = activation_enc(conv)

        conv = Conv2D(256, (4,4) ,padding='valid',kernel_initializer=init,kernel_regularizer=regularizers.l2(reg_enc),
          activity_regularizer=regularizers.l2(reg_enc))(conv)
        #conv26 = BatchNormalization(momentum=0.8)(conv25)
        conv = activation_enc(conv)

        #conv = Conv2D(256, tuple_conv,strides=1, padding='valid',kernel_initializer=init,kernel_regularizer=regularizers.l2(reg_enc),
         #activity_regularizer=regularizers.l2(reg_enc))(conv)
        #conv29 = BatchNormalization(momentum=0.8)(conv28)
        #conv = activation_enc(conv)

        flat = Flatten()(conv)

        d = Dense(1)(flat)


        modelcrit = Model(inputs =imgcrit ,outputs =  d)
        #validity = model(img)
        #model.summary()


        return modelcrit

    def _set_critic_graph(self):
        """
        Construct Computational Graph for the Critic
        The goal is to train the critic, without changing the generator.

        real_state -------------------------x----------------------|critic|--- real_score
                                            |
                                            |
        latent_state ----|generator*|---x--------------------------|critic|--- fake_score
                                        |   |
                                        |   |
                                        |---===|Rand.Weight.Ave.|--|critic|--- dummy

        generator* is not trainable.
        """

        # Freeze generator's layers while training critic
        self.generator.trainable = False

        # Image input (real sample)
        real_target = Input(shape=self.target_shape)

        # Noise input
        latent_state = Input(shape=(self.latent_dim,))

        # Generate image based of noise (fake sample)
        fake_target = self.generator(latent_state)

        # Discriminator determines validity of the real and fake
        fake_score = self.critic(fake_target)
        real_score = self.critic(real_target)

        # Construct weighted average between real and fake
        interpolated = RandomWeightedAverage(batch_size=self.batch_size)([real_target, fake_target])

        # Determine validity of weighted sample
        interpolated_score = self.critic(interpolated)

        # Use Python partial to provide loss function with additional
        # 'averaged_samples' argument
        partial_gp_loss = partial(self._gradient_penalty_loss,
                                  averaged_samples=interpolated)

        partial_gp_loss.__name__ = 'gradient_penalty'  # Keras requires function names

        self.critic_model = Model(inputs=[real_target, latent_state],
                                  outputs=[real_score, fake_score, interpolated_score])

        self.critic_model.compile(loss=[self._wasserstein_loss,
                                        self._wasserstein_loss,
                                        partial_gp_loss],
                                  optimizer=self.optimizerC,
                                  loss_weights=[1, 1, self.gradient_penalty])
        if self.summary:
            self.critic_model.summary()

    def _set_generator_graph(self):
        """
        Construct Computational Graph for Generator
        The goal is to train the generator, without changing the critic.

        latent_state ----|generator|-------|critic*|--- fake_score

        critic* is not trainable.
        """

        # For the generator we freeze the critic's layers
        self.critic.trainable = False
        self.generator.trainable = True

        # Sampled latent_states for input to generator
        latent_state = Input(shape=(self.latent_dim,))
        # Generate targets based on the latent_state
        fake_target = self.generator(latent_state)
        # Discriminator determines validity
        fake_score = self.critic(fake_target)

        # Defines generator model
        self.generator_model = Model(latent_state, fake_score)

        self.generator_model.compile(loss=self._wasserstein_loss,
                                     optimizer=self.optimizerG)

        if self.summary:
            self.generator_model.summary()

    def _gradient_penalty_loss(self, y_true, y_pred, averaged_samples):
        """Calculates the gradient penalty loss for a batch of "averaged" samples.
        In Improved WGANs, the 1-Lipschitz constraint is enforced by adding a term to the
        loss function that penalizes the network if the gradient norm moves away from 1.
        However, it is impossible to evaluate this function at all points in the input
        space. The compromise used in the paper is to choose random points on the lines
        between real and generated samples, and check the gradients at these points. Note
        that it is the gradient w.r.t. the input averaged samples, not the weights of the
        discriminator, that we're penalizing!
        In order to evaluate the gradients, we must first run samples through the generator
        and evaluate the loss. Then we get the gradients of the discriminator w.r.t. the
        input averaged samples. The l2 norm and penalty can then be calculated for this
        gradient.
        Note that this loss function requires the original averaged samples as input, but
        Keras only supports passing y_true and y_pred to loss functions. To get around this,
        we make a partial() of the function with the averaged_samples argument, and use that
        for model training."""
        # first get the gradients:
        #   assuming: - that y_pred has dimensions (batch_size, 1)
        #             - averaged_samples has dimensions (batch_size, nbr_features)
        # gradients afterwards has dimension (batch_size, nbr_features), basically
        # a list of nbr_features-dimensional gradient vectors
        gradients = K.gradients(y_pred, averaged_samples)[0]
        # compute the euclidean norm by squaring ...
        gradients_sqr = K.square(gradients)
        #   ... summing over the rows ...
        gradients_sqr_sum = K.sum(gradients_sqr,
                                  axis=np.arange(1, len(gradients_sqr.shape)), keepdims = True)
        #   ... and sqrt
        gradient_l2_norm = K.sqrt(gradients_sqr_sum)
        # compute lambda * (1 - ||grad||)^2 still for each single sample
        gradient_penalty = K.square(1 - gradient_l2_norm)
        # return the mean as loss over all the batch samples
        return K.mean(gradient_penalty, keepdims = True)

    def _wasserstein_loss(self, y_true, y_pred):
        """Calculates the Wasserstein loss for a sample batch.
        The Wasserstein loss function is very simple to calculate. In a standard GAN, the
        discriminator has a sigmoid output, representing the probability that samples are
        real or generated. In Wasserstein GANs, however, the output is linear with no
        activation function! Instead of being constrained to [0, 1], the discriminator wants
        to make the distance between its output for real and generated samples as
        large as possible.
        The most natural way to achieve this is to label generated samples -1 and real
        samples 1, instead of the 0 and 1 used in normal GANs, so that multiplying the
        outputs by the labels will give you the loss immediately.
        Note that the nature of this loss means that it can be (and frequently will be)
        less than 0."""
        return K.mean(y_true * y_pred)

    def _targets_generator(self, **kwargs):
        """
        Return a batch of real data of size:
        self.batch_size * self.n_critic
        """
        while True:
            idx = np.random.randint(0, self.data.shape[0], self.batch_size * self.n_critic)
            real_targets = self.data[idx]
            yield real_targets

    def _train_learning_rate(self, epoch, d_loss, g_loss):
        if epoch % 3000 == 0 and epoch > 0:  # Learning rate modification
            initial_lrate = K.get_value(self.critic_model.optimizer.lr)
            lrate = initial_lrate * 0.7
            K.set_value(self.critic_model.optimizer.lr, lrate)
            initial_lrate = K.get_value(self.generator_model.optimizer.lr)
            lrate = initial_lrate * 0.7
            K.set_value(self.generator_model.optimizer.lr, lrate)

    def train(self, epochs, save_interval=None, save_file='name_save', run_number = '-1', log_file='wgan.log', log_interval=10, data_generator=None, save_intermediate_model=False, **kwargs):

        self.history = { 'generator': [],
                     'critic': [],
                     'wasserstein': [] }

        # Load the dataset
        # X_train = self._load_data(**kwargs) #dataExtraction_puma(morph=False)
        self.run_number = run_number
        # Adversarial ground truths
        positive_ones = np.ones((self.batch_size, 1))
        negative_ones = -positive_ones
        dummy = np.zeros((self.batch_size, 1))  # Dummy gt for gradient penalty

        save_interval = epochs//10 if save_interval is None else save_interval

        if data_generator is None:
            data_generator = self._targets_generator()

        log_dir = self.create_logdir()

        #if ~os.path.exists(os.path.dirname(log_file)):
            #os.makedirs()

        loss_hist_g = np.zeros((epochs,1))
        loss_hist_d = np.zeros((epochs,4))

        with open(log_file, 'w') as log_file:

            self.fakes = {}

            for epoch, real_targets_minibatches in zip(range(self.first_epoch,epochs), data_generator):

                

                # ---------------------
                # -1- Compute the Wasserstein distance by maximizing the critic
                #     i.e. Train Discriminator
                # ---------------------
                for k in range(self.n_critic):
                    # Load real targets extracted from the sample of real targets.
                    real_targets = real_targets_minibatches[k * self.batch_size:(1 + k) * self.batch_size]

                    # Sample generator input
                    latent_samples = np.random.normal(size=(self.batch_size, self.latent_dim))

                    #print('Before d_loss')
                    # Train the critic
                    d_loss = self.critic_model.train_on_batch(
                                            [real_targets, latent_samples],
                                            [positive_ones, negative_ones, dummy])
                            # [negative_ones, positive_ones, dummy]) --> formulation article
                    #print('After d_loss')
                # ---------------------
                # -2- Minmizing the Wasserstein distance
                #     i.e. Train Generator
                # ---------------------

                # Train from the last generated `latent_samples` & `real_valid`
                # The objective is to force the model to produce realistic outputs:
                # The critics of the generated samples should be $1$ meaning the critic considers
                # the generated states as realistic ones.
                #
                #print('Before g_loss')
                g_loss = self.generator_model.train_on_batch(latent_samples,
                                                         positive_ones)
                                                        # negative_ones) ---> formulation article


                #tfboard = False
                
                print (str(epoch)+" [D loss: "+str(d_loss)+"] [G loss: "+str(g_loss)+"] [Wasserstein distance" +str(d_loss[1]-g_loss)+"]" )
                loss_hist_g[epoch,0]=g_loss
                loss_hist_d[epoch,:]=d_loss[0:5]

                if epoch % log_interval == 0:
                    if self.tfboard :
                        loss_summary = tf.Summary(value=[tf.Summary.Value(tag="Wasserstein_loss", 
                                                         simple_value=d_loss[1]-g_loss), ])
                        #lr_summary = tf.Summary(value=[tf.Summary.Value(tag="lr", 
                        #                                 simple_value=self.critic_model.optimizer.lr.eval()), ])
            
                        #g_img = self.generator.predict(noise)
                        #image_summary = tf.Summary(value=[tf.Summary.Image(tag="first gen", 
                        #                                 Image=g_img[0,:,:,0]), ])
                        #log_dir.add_summary([loss_summary] , epoch)
                        self._train_learning_rate(epoch, d_loss, g_loss)
        
                        # Diagnosis of models & intermediate saving
        
                        self.history['generator'].append(g_loss)
                        self.history['critic'].append(d_loss)
                        wasserstein = -d_loss[0] - d_loss[1]
                        self.history['wasserstein'].append(wasserstein)
                    

                if epoch % save_interval == 0 or epoch==epochs-1:
                    fakes = self.generator.predict(latent_samples[:10])
                    out_file = save_file    
                    self.fakes[epoch] = fakes
                    self.save_fakes(out_file, fakes,epoch)
                    if save_intermediate_model:
                        self.save_model_cam(out_file)
                np.save( './log_dir/run'+str(self.run_number)+'/loss_hist_d',loss_hist_d)
                np.save( './log_dir/run'+str(self.run_number)+'/loss_hist_g',loss_hist_g)
        return

    def create_logdir(self):
        import shutil
        import tensorflow as tf
        my_log_dir = './log_dir/run'+str(self.run_number)+'/'
        train_log_dir = os.path.join(my_log_dir, 'training/')
        val_log_dir = os.path.join(my_log_dir, 'validation/')
        if os.path.exists(os.path.dirname(my_log_dir)):
            response = input(my_log_dir + " already exists ! 'del' or 'stop' ? ")
            
            if response == 'del':
                shutil.rmtree(my_log_dir)
            else:
                exit()
        else:
            os.makedirs(os.path.dirname(my_log_dir))
            os.makedirs(os.path.dirname(train_log_dir))
            os.makedirs(os.path.dirname(val_log_dir))

        save_dir = './save_model/run'+str(self.run_number)+'/'
        if os.path.exists(os.path.dirname(save_dir)):
            for file in os.listdir(save_dir):
                file_path = os.path.join(save_dir, file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(e)
        else:
            os.makedirs(os.path.dirname(save_dir))

        training_log_dir = tf.summary.FileWriter(train_log_dir)
        return training_log_dir

    def save_weights_cam(self, base_name):
        '''Function saving the weights of the GAN
        '''
        if not os.path.exists('./weights'):
            os.makedirs('weights')
        self.critic.save_weights('./weights/'+ base_name + '_critic.h5')
        self.generator.save_weights('./weights/'+ base_name + '_generator.h5')

    def save_model_cam(self, base_name):
        '''Function saving the weights of the GAN
        '''
        if not os.path.exists('./model'):
            os.makedirs('model')
        self.critic.save('./model/'+ base_name + '_critic.h5')
        self.generator.save('./model/'+ base_name + '_generator.h5')

    def save_fakes(self, base_name, fakes, epoch):
        '''Function saving the weights of the GAN
        '''
        if not os.path.exists('./data/generated/wgan/'+base_name+'/'):
            os.makedirs('./data/generated/wgan/'+base_name+'/')
        np.save('./data/generated/wgan/'+ base_name +'/'+base_name + '_'+str(epoch)+ '.npy', fakes)

    def load_weights_cam(self, base_name):
        '''Function loading the weights of the GAN
        '''
        print('finally !!')
        self.critic.load_weights( base_name + '_critic.h5')
        self.generator.load_weights(base_name + '_generator.h5')
