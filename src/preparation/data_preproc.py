


import numpy as np
import h5py as h5
import copy as cp


def dataExtraction_puma(DB_path='', DB_name = '', im_shape = (64,128,5)):
    '''
    Data extraction sum of sizes has to be inferior to DB_size/3.
    Flages : pattern : -1 => CHAN, 0 => FAULT, 1 => LOBE 
    2 different dataset original one and one with morphology operation applied (not working)
    '''
    rows = im_shape[0]
    cols = im_shape[1]
    chans = im_shape[2]
    scaling = np.zeros((chans, 2))  # contain mean and var for the 4 chans
    DB_size = 3*365
    print('Size of the DB : ',DB_size)
    DB_images=np.ones((DB_size, rows, cols, chans))

    f=h5.File(DB_path,'r')
    
    f[DB_name].read_direct(DB_images)

    print('Dataset used : 1095 images 81 channels - 3y simulation')
    for chan in range(chans):
        DB_images[:,:,:,chan],scaling[chan,0],scaling[chan,1] = scale2(DB_images[:,:,:,chan])
        inter = DB_images[:,:,:,chan]
    f.close()

    return DB_images,scaling


def dataPreparationPuma(DB_path='', DB_name = '', im_shape = (64,128,81)):
    '''
    '''
    rows = im_shape[0]
    cols = im_shape[1]
    chans = im_shape[2]
    scaling = np.zeros((chans,2)) #contain mean and var for the 4 chans
    DB_size=3*365
    print('Size of the DB : ',DB_size)
    DB_images=np.ones((DB_size,rows,cols,chans))
    
    #f=h5.File('./DB/data_puma_50y_ta-ua-va-zeta-ps.h5','r') #Eternal january puma 
    f=h5.File(DB_path,'r')
    
    #f['data_puma_v1'].read_direct(DB_images)
    f[DB_name].read_direct(DB_images)
    print('Dataset used : k images 10 channels - 50y simulation')
    for chan in range(chans):
        DB_images[:,:,:,chan], scaling[chan,0], scaling[chan,1] = scale2(DB_images[:,:,:,chan])
        inter = DB_images[:,:,:,chan]
        #scaling[chan,0] = np.mean(inter,axis = 0)
        #scaling[chan,1] = np.std(inter,axis = 0)**2
    x_trainf=cp.copy(DB_images)
    print('saving...')
    with h5.File('../../data/raw/data_plasim_3y_sc.h5', 'w') as f:
        dset = f.create_dataset("dataset", data = x_trainf)
        dset = f.create_dataset("scaling", data = scaling)
        print(list(f.keys()))
        #dset = f['mydataset']
        #print(dset.shape)
    print('saving done')
    f.close()

    return 


def scale2(X):
    mean = np.mean(X)
    X = (X[:]-mean)
    m = np.std(X)
    X1 = X[:]/m

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



#dataExtraction_puma(DB_path='../../data/raw/data_plasim_3y.h5', DB_name='dataset')