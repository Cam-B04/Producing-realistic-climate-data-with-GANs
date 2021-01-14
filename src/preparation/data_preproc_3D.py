


import numpy as np
import h5py as h5
import copy as cp

def dataExtraction_puma_3D(DB_path='', DB_name = '', im_shape = (64,128,5,4)):
    '''
    Data extraction sum of sizes has to be inferior to DB_size/3.
    Flages : pattern : -1 => CHAN, 0 => FAULT, 1 => LOBE 
    2 different dataset original one and one with morphology operation applied (not working)
    '''
    scaling = np.zeros((20,2)) #contain mean and var for the 4 chans
    DB_size=36524
    print('Size of the DB : ',DB_size)
    _DB_images=np.ones((DB_size,64,128,21))
    DB_images=np.ones((DB_size-1000,64,128,21))
    
    #f=h5.File('./DB/data_puma_50y_ta-ua-va-zeta-ps.h5','r') #Eternal january puma 
    f=h5.File('../Dataset/simu100y/100yPlasim_21chans.h5','r')
    
    #f['data_puma_v1'].read_direct(DB_images)
    f['100yPlasim_21chans'].read_direct(_DB_images)
    #np.swapaxes(DB_images,0,3)
    #DB_images = DB_images.reshape(DB_size,64,128,4)
    #DB_images = np.swapaxes(DB_images,0,1)
    #DB_images = np.swapaxes(DB_images,1,2)
    #DB_images = np.swapaxes(DB_images,2,3)
    #print(DB_images.shape)
    DB_images = _DB_images[1000:,:,:,:20]
    print(DB_images.shape)
    print('Dataset used : 36k images 4 channels - 100y simulation')
    for chan in range(20):
        DB_images[:,:,:,chan],scaling[chan,0],scaling[chan,1] = scale2(DB_images[:,:,:,chan])
        inter = DB_images[:,:,:,chan]
        #scaling[chan,0] = np.mean(inter,axis = 0)
        #scaling[chan,1] = np.std(inter,axis = 0)**2
    x_trainf=cp.copy(DB_images)
    f.close()
    x_trainf = x_trainf.reshape(35524,64,128,5,4)
    scaling = scaling.reshape(5,4,2)

    np.save('./data/raw/3D_save_X_train', x_trainf)
    np.save('./data/raw/3D_save_X_scaling', scaling)

    print('save DONE !')
    return x_trainf,scaling


def scale2(X):
    print(X.shape)
    mean = np.mean(X)
    print('mean:',mean)
    X=(X[:]-mean)
    m = np.std(X)
    print('m :',m)
    X1=X[:]/m
    #if np.max(np.abs(X1))>1.01 :
        #print('mauvaise scaling!!!!!!!!')
    return X1,mean,m


def plotCam(data,titletxt,xlabel,ylabel,labels,N,savename):
    '''
    Plotting function for max N<=5 curves
    '''
    colors=['k','0.80','0.60','0.40','0.30']
    style=['--','-','-.',':','--']
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize=40)
    plt.rc('ytick', labelsize=40)

    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(1, 1, 1,title=titletxt)

    for i in range(N):
        ax.plot(savgol_filter(data[:,i], 51, 3), color=colors[i], ls=style[i], label=labels[i],linewidth=0.5)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)


    y = data[:,0]-data[:,1]
    yhat = savgol_filter(y, 51, 3)

    ax.plot(yhat, color=colors[i+1], ls=style[i+1], label=labels[i+1],linewidth=0.5)
    ax.legend()

    plt.tight_layout()
    #plt.show()
    plt.savefig(savename)
    plt.close()
