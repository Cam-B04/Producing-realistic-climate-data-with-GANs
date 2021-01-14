
##############Memory managment###############
import tensorflow as tf

#Setting for memory allocaton of the GPU. 
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

sess = tf.Session()

import numpy as np
import numpy.random as nr
import numpy as np

import keras


import keras.layers.merge
from keras.layers import Layer
from keras.layers.merge import _Merge
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D

from keras.layers import Input, Lambda, Layer, Dense, Reshape,BatchNormalization,Input,Dropout ,Dense, Lambda, Conv2D, MaxPooling2D, UpSampling2D, Flatten,Conv2DTranspose,Activation, Cropping2D
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, BatchNormalization, Activation, ZeroPadding2D
from keras.layers import Reshape,BatchNormalization,Input,Dropout ,Dense, Lambda, Conv2D, MaxPooling2D, UpSampling2D, Flatten,Conv2DTranspose,Activation

from keras.models import Sequential, Model,load_model


from keras.optimizers import RMSprop, Adam, SGD
from keras import regularizers
from keras.initializers import RandomNormal

from functools import partial
from keras.models import Model
from keras import backend as K
from keras import metrics,regularizers
from keras.losses import mse, binary_crossentropy, kullback_leibler_divergence,mae

import sys
sys.path.append('./src/modeling')

sys.path.append('./src/preprocessing')

sys.path.append('./src/preparation')

from wgan_gp_V6 import *
from data_preproc import *
#from data_prepa import *  

def mask_man(Np,width):
    line,col = 100,100
    width_list = [x for x in range(-width,width+1)]
    masks = np.zeros((line,col,1))
    masks_batch = np.zeros((_bs ,line,col,1))

    index = 40

    masks[10,10]=1.
    masks[90,10]=1.
    masks[10,90]=1.
    masks[90,90]=1.
    masks[50,50]=1.

    samp_ind = np.isin(masks, 1.)
    for i in width_list:
        for j in width_list:
            masks[samp_ind[:,0]+i, samp_ind[:,1]+j ] = 1./(float(np.abs(j)+np.abs(i))+1)**2 
    #masks[ 0, samp_ind[:,0]+i, samp_ind[:,1]+j, 0 ] = [ samp_pnts.reshape(Np, )/float(np.abs(j+i+1)) for i,j in [-3,-2,-1 , 1 ,2 ,3]]
    
    masks_batch[:] = masks

    return masks_batch

def mask_exp2_puma(arr, Np,width, sample_mod = ''):
    line,col, chans = arr.shape[1],arr.shape[2], arr.shape[3]
    width_list = [x for x in range(-width,width+1)]
    masks = np.zeros((line,col,chans))
    masks_batch = np.zeros((_bs ,line,col,chans))
    bmask = np.zeros((_bs, line,col,chans))
    #batch = np.random.randint(0,arr.shape[0]-1,(batch_size,))
    imgs = cp.copy(arr)
    #ind = np.random.randint(0,10000)
    if sample_mod=='europe':
        samp_ind1 = np.random.randint(10,29,(Np,))
        samp_ind2 = np.random.randint(57,82,(Np,))
    else:
        samp_ind1 = np.random.randint(0+width,line-width,(Np,))
        samp_ind2 = np.random.randint(0+width,col-width,(Np,))
    #stack les deux samp ind
    samp_ind = np.stack([samp_ind1,samp_ind2],axis=1)
    #print(samp_ind.shape)
    samp_pnts = imgs[0,samp_ind[:,0],samp_ind[:,1],:]

    masks[samp_ind[:,0],samp_ind[:,1],:] = np.squeeze(samp_pnts.reshape(Np,chans))
    bmask[:,samp_ind[:,0],samp_ind[:,1],:] = 1.
    #print(np.squeeze(samp_pnts.reshape(Np,5)).shape)
    for i in width_list:
        for j in width_list:
            masks[samp_ind[:,0]+i, samp_ind[:,1]+j,: ] = samp_pnts.reshape(Np,chans)/(float(np.abs(j)+np.abs(i))+1)**2
    #masks[ 0, samp_ind[:,0]+i, samp_ind[:,1]+j, 0 ] = [ samp_pnts.reshape(Np, )/float(np.abs(j+i+1)) for i,j in [-3,-2,-1 , 1 ,2 ,3]]
    
    masks_batch[:] = masks

    return masks_batch, samp_ind, tf.cast(tf.convert_to_tensor(bmask),dtype = tf.float32)
'''#ind = np.random.randint(0,10000)
samp_ind = np.random.randint(0+5,line-5,(Np,2))
samp_pnts = imgs[0,samp_ind[:,0],samp_ind[:,1],:]
masks[samp_ind[:,0],samp_ind[:,1],:] = samp_pnts.reshape(Np,1)
for i in width_list:
    for j in width_list:
        masks[samp_ind[:,0]+i, samp_ind[:,1]+j ] = samp_pnts.reshape(Np,1)/(float(np.abs(j)+np.abs(i))+1)**2
#masks[ 0, samp_ind[:,0]+i, samp_ind[:,1]+j, 0 ] = [ samp_pnts.reshape(Np, )/float(np.abs(j+i+1)) for i,j in [-3,-2,-1 , 1 ,2 ,3]]

masks_batch[:] = masks'''
#return masks_batch

class PB_layer_pad(Layer):

    def __init__(self,img_shape=(64,128,5), axis=2, padding=2, **kwargs):
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

class INN(object):
    """
    Implementation of the inference neural network inspired by
    https://arxiv.org/pdf/1807.05207.pdf S.Chan A.H. Elsheikh
    
    """
    def __init__(self, input_dim=128, output_dim = 32, layers_num = 3, layers_size = 256, batch_size=32,
                 optimizer=None, summary=False,models=None, generator = None, tfboard = False, b_mask = None):
    

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layers_num = layers_num
        self.layers_size = layers_size
        if b_mask is None:
            print('No bmask defined!!!') 
        else :
            self.b_mask = b_mask

        self.batch_size = batch_size

        if optimizer is None:
            self.optimizer = Adam(lr=0.0002, beta_1=0.5, beta_2=0.999, epsilon=None, decay=0.00001, amsgrad=False)
    
      
        self.tfboard = tfboard
        self.summary = summary

        self.inn = self._build_inn(neurons = self.layers_size)


        if self.summary:
            self.inn.summary()

        self.generator = generator

        if self.generator != None: 
            self._set_inn_graph()






    def _build_inn(self, neurons = 256):
        acti = LeakyReLU(alpha=0.5)
        reg = 0.0001
        w = Input(shape=(self.input_dim,))
        layer = Dense(neurons,kernel_regularizer=regularizers.l2(reg),kernel_initializer = RandomNormal(mean=0.0, stddev=0.02, seed=None))(w)
        layer = acti(layer)
        #layer = BatchNormalization(momentum=0.99)(layer)
        for i in range(self.layers_num-1):
            layer = Dense(neurons,kernel_regularizer=regularizers.l2(reg),kernel_initializer = RandomNormal(mean=0.0, stddev=0.02, seed=None))(layer)
            layer = acti(layer)
            #layer = BatchNormalization(momentum=0.99)(layer)

        z = Dense(self.output_dim,kernel_regularizer=regularizers.l2(reg),kernel_initializer = RandomNormal(mean=0.0, stddev=0.02, seed=None))(layer)
        

        return Model(w,z)


    def _set_inn_graph(self):
        self.generator.trainable = False


        _w = Input(shape = (self.input_dim,))
        _z = self.inn(_w)
        _img = self.generator(_z)

        self.cgan = Model(inputs = _w, outputs = [_z,_z,_img])
        loss_inn = self.masked_loss_inn(self.b_mask)
        self.cgan.compile(loss = [self.loss_norm, self.entropy_loss, loss_inn],
         optimizer = self.optimizer, loss_weights = [10000.0, 1000., 10000.0])

        if self.summary:
            self.cgan.summary()


    def loss_norm(self,y_true, y_pred):
        return tf.abs(tf.norm(y_pred,axis = 1)-tf.sqrt(tf.cast(self.output_dim, dtype = tf.float32)))

    def entropy_loss(self, y_true, y_pred):
        def squared_dist(A): 
            expanded_a = tf.expand_dims(A, 1)
            expanded_b = tf.expand_dims(A, 0)
            distances = tf.reduce_sum(tf.squared_difference(expanded_a, expanded_b), 2)
            return distances
        #pri    nt(K.shape(X_p))
        _bs = K.variable(self.batch_size)
        k_t = K.round(K.sqrt(_bs))
        k_t = tf.cast(k_t,tf.int32)
        neg_one = tf.constant(-1.0, dtype=tf.float32)
        # we compute the L-1 distance
        #distances =  tf.reduce_sum(tf.abs(tf.subtract(X_t, x_t)), 1)
        distances = squared_dist(y_pred)
        # to find the nearest points, we find the farthest points based on negative distances
        # we need this trick because tensorflow has top_k api and no closest_k or reverse=True api
        neg_distances = tf.multiply(distances, neg_one)
        # get the indices
        vals, indx = tf.nn.top_k(neg_distances, k_t)
        # slice the labels of these points
        y_s = tf.reduce_mean(tf.log(tf.abs(vals)+K.epsilon()))
        dim = K.shape(y_pred)
        y_s = tf.multiply(tf.multiply(y_s,tf.cast(dim[1],tf.float32)),neg_one)
        #tf.cast(dim[1],dtype = tf.float32)*
        return y_s
    def masked_loss_inn(self, bmask):
        def loss_inn( y_true, y_pred):
            return K.sum( tf.multiply( 
                                        tf.square( tf.add(y_pred,-y_true) )
                                                                     ,tf.abs(bmask) ),axis=(1,2))
        return loss_inn

    def _train_learning_rate(self, epoch, d_loss, g_loss):
        if epoch % 3000 == 0 and epoch > 5000:  # Learning rate modification
            initial_lrate = K.get_value(self.critic_model.optimizer.lr)
            lrate = initial_lrate * 0.7
            K.set_value(self.critic_model.optimizer.lr, lrate)
            initial_lrate = K.get_value(self.generator_model.optimizer.lr)
            lrate = initial_lrate * 0.7
            K.set_value(self.generator_model.optimizer.lr, lrate)

    def train(self, epochs, inference=None, save_file='name_save', run_number = '-1', log_file='cgan.log', log_interval=10, mask=None, save_intermediate_model=False, **kwargs):
        # Load the dataset
        # X_train = self._load_data(**kwargs) #dataExtraction_puma(morph=False)
        #self.run_number = run_number

        save_interval = epochs//10 



        #log_dir = self.create_logdir()

        #if ~os.path.exists(os.path.dirname(log_file)):
            #os.makedirs()
        loss_hist = np.zeros((epochs,4))
        start_time = datetime.datetime.now()
        for epoch in range(epochs):
            # ---------------------
            #  Train Inference Network
            # ---------------------
            noise = np.random.normal(0, 1, (self.batch_size,self.input_dim))
            loss = self.cgan.train_on_batch([noise], [noise,noise,mask])
            elapsed_time = datetime.datetime.now() - start_time
            print ("[Epoch %d/%d] [Inn avg loss: %f, norm %f, entrop est %f, conditioning loss: %f] time: %s" % (epoch, epochs,
                                                                    loss[0],
                                                                    loss[1],
                                                                    loss[2],
                                                                    loss[3],
                                                                    elapsed_time))
            loss_hist[epoch,:]=loss[:4]

            if inference and (epoch)%500==0:
                    self.sample_images(epoch, self.cgan, mask, save_file)
            if (epoch+1)%10000 == 0 and epoch != 0:
                    self.save_model_cam(save_file)
        np.save('./loss_hist', loss_hist)
        plt.figure()
        plt.plot(loss_hist)
        plt.savefig('./loss_hist.pdf')
        return


    def sample_images(self, epoch, model ,mask ,path):
        image_path = path
        #os.makedirs(image_path , exist_ok=True)
        r, c = 3, 3
        #imgs_A, imgs_B = self.data_loader.load_data(batch_size=3, is_testing=True)
        #ind = np.random.randint(0,10000,(c,))
        #im_A = cp.copy(trainset[ind])
        
        noise = np.random.normal(0, 1, (r*c,self.input_dim,))
        z,_z,imgs = self.cgan.predict(noise)
        
        # Rescale images 0 - 1
        #gen_imgs = 0.5 * gen_imgs + 0.5
        #imgs_B_mask = np.zeros((imgs_B.shape))
        #print(imgs_B.shape)
        #titles = ['Condition', 'Generated', 'Original']
        fig, axs = plt.subplots(r, c)
        cnt = 0

        for i in range(r):
            for j in range(c):
                im = axs[i,j].imshow(imgs[cnt,:,:,0])
                
                #axs[i,j].imshow(imgs_B_mask0)
                #axs[i,j].imshow(imgs_B_mask1[j,:,:,0],cmap='gray', alpha=0.3)
                scat1 = np.where(mask[0,:,:,0] == 1.)
                scat0 = np.where(mask[0,:,:,0] == -1.)
                #scat = np.where(imgs_B_mask)
                
                axs[i, j].scatter(scat1[1][:],scat1[0][:],marker = '.',s=12)
                axs[i, j].scatter(scat0[1][:],scat0[0][:],marker ='x', s=12)
                
                #plt.colorbar(axs[i, j])
                axs[i,j].axis('off')
                cnt += 1

        fig.subplots_adjust(right=0.8)
        plt.suptitle(' Conditioned generations with inference network after'+str(epoch)+'epochs')
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(im, cax=cbar_ax)


        if not os.path.exists(image_path):
            os.makedirs('./' + image_path)
        fig.savefig( image_path + "inn_epoch_%d.png" % (epoch))
        plt.close()
        print(z[0])


    def save_model_cam(self, base_name):
        '''Function saving the weights of the GAN
        '''
        if not os.path.exists('./model/inn/'):
            os.makedirs('model/inn/')
        self.cgan.save('./model/inn/'+ base_name + '_inn.h5')
        


if __name__ == '__main__':

    _bs = 128

    gen = load_model('./model/name_save_fakes_1500_generator.h5',custom_objects={'PB_layer_pad': PB_layer_pad}) 
    gen.name = 'generator_model'

    X_train, scaling = dataExtraction_puma(DB_path='./data/raw/100yPlasim/100yPlasim.h5',
                 DB_name = '100yPlasim', im_shape = (64,128,5))
    #X_train = (X_train-1.5)*2.
    #X_train = tf.cast(X_train, dtype =tf.float32)
    X_train.astype(np.float32)

    Np = 10;
    extp = 2;

    ind = np.random.randint(0,10000,(_bs,)) 
    imgs_A = cp.copy(X_train[ind,:,:,:])
    imgs_B = np.zeros((imgs_A.shape))
    #imgs_B[:,:,:] = mask_man(Np,extp)
    imgs_B[:,:,:],samp_index, bmask = mask_exp2_puma(imgs_A,Np,extp)

    cgan = INN(input_dim=8, output_dim = 32, layers_num = 5, layers_size = 256, batch_size=_bs,
                 optimizer=None, summary=True, models=None,
                 generator = gen, tfboard = False, b_mask = bmask)  


    cgan.train(epochs = 5000, inference=True, save_file='./data/generated/inn/inn_train',run_number = '-1', 
                log_file='cgan.log', log_interval=10, mask=imgs_B, save_intermediate_model=False)



















