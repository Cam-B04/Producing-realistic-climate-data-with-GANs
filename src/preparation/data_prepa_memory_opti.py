
import h5py as h5 
import h5netcdf
import netCDF4
import numpy as np



#with h5netcdf.File('./mydata100yplasim.nc', 'r') as f:
class HDF5Store(object):
    """
    Simple class to append value to a hdf5 file on disc (usefull for building keras datasets)
    
    Params:
        datapath: filepath of h5 file
        dataset: dataset name within the file
        shape: dataset shape (not counting main/batch axis)
        dtype: numpy dtype
    
    Usage:
        hdf5_store = HDF5Store('/tmp/hdf5_store.h5','X', shape=(20,20,3))
        x = np.random.random(hdf5_store.shape)
        hdf5_store.append(x)
        hdf5_store.append(x)
        
    From https://gist.github.com/wassname/a0a75f133831eed1113d052c67cf8633
    """
    def __init__(self, datapath, dataset, shape, dtype=np.float32, compression="gzip", chunk_len=1):
        self.datapath = datapath
        self.dataset = dataset
        self.shape = shape
        self.i = 0
        
        with h5.File(self.datapath, mode='w') as h5f:
            self.dset = h5f.create_dataset(
                dataset,
                shape=(0, ) + shape,
                maxshape=(None, ) + shape,
                dtype=dtype,
                compression=compression,
                chunks=(chunk_len, ) + shape)
    
    def append(self, values):
        with h5.File(self.datapath, mode='a') as h5f:
            dset = h5f[self.dataset]
            dset.resize((self.i + 1, ) + shape)
            dset[self.i] = [values]
            self.i += 1
            h5f.flush()
def process_ncdf_file(chans ):
	L = ['ta','ua','va','zeta','ps']
	data = np.zeros((36524,21, 64, 128))
	fn = netCDF4.Dataset('./mydata100yplasim.nc', 'r')
	print(fn)
	print(fn['ta'])
	print(fn.variables['ta'])
	print(fn.variables.keys())
	for i in range(5):
		if i==4:
			data[:,i*chans,:,:] = fn.variables[L[i]][:,:,:]
		else:
			data[:,i*chans:(i+1)*chans,:,:] = fn.variables[L[i]][:,:chans,:,:]
		print('Variables ',i,'/5... ', L[i])
	
	data = np.swapaxes(data,1,2)
	data = np.swapaxes(data,2,3)
	#np.swapaxes(data,3,4)
	with h5.File('100yPlasim_21chans.h5', 'w') as f:
		dset = f.create_dataset("100yPlasim_21chans", data = data)
		
		#list(f.keys())
		#dset = f['mydataset']
		#print(dset.shape)
	return


'''fn3 = netCDF4.Dataset('../../data/raw/100y_10layers.nc', 'r')
rootgrp = netCDF4.Dataset('../../data/raw/100y_10layers.h5', 'w', format = 'NETCDF4')
#fullSimGrp = rootgrp.createGroup('fullSim')
print('starting...')
rootgrp = fn3.variables
print(fn3.variables)
fn3.close()
#rootgrp.close()
print('Done.')

shape = (15000,81,64,128)
shift = int(36524*0.5)

chans = 10
'''

fn = h5.File('../../data/raw/100y_10layers.h5', 'r')
#fn3 = netCDF4.Dataset('../../data/raw/100y_10layers.h5', 'r')
L3 = list(fn3.keys())
print(L3)
L = list(fn.keys())
print(L)
print(fn.keys())
print(fn)
#print(fn['ta'])
for i in L:
    print(i)


'''for t in range(6):

	data = np.zeros((5000,81, 64, 128))
	
	
	for i in range(4,12):
	    if i==10:
	    	i=11
	    	data[:,6*chans:7*chans,:,:] = fn.variables[L[12]][6524+5000*t:6524+(t+1)*5000,:chans,:,:]
	        #data[:,(i-4)*chans,:,:] = fn.variables[L[i]][:30000,:,:]
	    else:
	        data[:,(i-4)*chans:(i+1-4)*chans,:,:] = fn.variables[L[i]][6524+5000*t:6524+(t+1)*5000,:chans,:,:]
	    print('Variables ',i,'/5... ', L[i])
	#data[:,(i-4)*chans:(i+1-4)*chans,:,:] = fn.variables[L[i]][:30000,:chans,:,:]
	data[:,-1,:,:] = fn.variables[L[10]][6524+5000*t:6524+(t+1)*5000,:,:]
	print('data is fully created.')
	print('Swapping axis...')
	data = np.swapaxes(data,1,2)
	data = np.swapaxes(data,2,3)
	print('done.')
	#np.swapaxes(data,3,4)
	print('saving .h5 file.')
	#np.save('../../data/raw/100yPlasim_81chans.h5', data)
	print('data saved .npy file')
	
	if i==0:
		hdf5_store = HDF5Store('../../data/raw/100yPlasim_81chans_chunked.h5','X', shape=shape)
	else:
		hdf5_store.append(data)
   '''
data = fn[:]
'''

for i in range(4,12):
    if i==10:
    	i=11
    	data[:,6*chans:7*chans,:,:] = fn.variables[L[12]][15000+6524:,:chans,:,:]
        #data[:,(i-4)*chans,:,:] = fn.variables[L[i]][:30000,:,:]
    else:
        data[:,(i-4)*chans:(i+1-4)*chans,:,:] = fn.variables[L[i]][15000+6524:,:chans,:,:]
    print('Variables ',i,'/5... ', L[i])
#data[:,(i-4)*chans:(i+1-4)*chans,:,:] = fn.variables[L[i]][:30000,:chans,:,:]
data[:,-1,:,:] = fn.variables[L[10]][6524+15000:,:,:]'''
print('data is fully created.')
print('Swapping axis...')
data = np.swapaxes(data,1,2)
data = np.swapaxes(data,2,3)
print('done.')
'''
#np.swapaxes(data,3,4)
print('saving .h5 file.')
#np.save('../../data/raw/100yPlasim_81chans.h5', data)
print('data saved .npy file')
'''



 
# test



