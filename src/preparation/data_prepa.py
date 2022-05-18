
import h5py as h5 
import h5netcdf
import netCDF4
import numpy as np


nc_file = '../../../../PLASIM/postprocessor/data_plasim_50.nc'
#with h5netcdf.File('./mydata100yplasim.nc', 'r') as f:

def process_ncdf_file(chans ):
    L = ['ta', 'ua', 'va', 'wap', 'zeta', 'pl', 'd', 'zg', 'hur']
    data = np.zeros((18262,10, 64, 128))
    fn = netCDF4.Dataset(nc_file, 'r')
    print(fn)
    print('starting...')
    for i in range(len(L)):
        if i==4:
            data[:,i*chans,:,:] = fn.variables[L[i]][:,:,:]
        else:
            data[:,i*chans:(i+1)*chans,:,:] = fn.variables[L[i]][:,:chans,:,:]
        print('Variables ',i,'/5... ', L[i])

    data = np.swapaxes(data,1,2)
    data = np.swapaxes(data,2,3)
    #np.swapaxes(data,3,4)
    with h5.File('data_plasim_50.h5', 'w') as f:
        dset = f.create_dataset("50yPlasim_10lay", data = data)
        
        #list(f.keys())
        #dset = f['mydataset']
        #print(dset.shape)
    return

process_ncdf_file(10)

'''chans = 10
fn = netCDF4.Dataset(nc_file, 'r')
L = list(fn.variables.keys())
print(fn.variables.keys())
print(fn)
#print(fn['ta'])
for i in L:
    print(fn.variables[i])

samples = 18262
data = np.zeros((samples,81, 64, 128))
for i in range(4,12):
    if i==10:
    	i=11
    	data[:,6*chans:7*chans,:,:] = fn.variables[L[12]][:samples,:chans,:,:]
        #data[:,(i-4)*chans,:,:] = fn.variables[L[i]][:30000,:,:]
    else:
        data[:,(i-4)*chans:(i+1-4)*chans,:,:] = fn.variables[L[i]][:samples,:chans,:,:]
    print('Variables ',i,'/5... ', L[i])
#data[:,(i-4)*chans:(i+1-4)*chans,:,:] = fn.variables[L[i]][:30000,:chans,:,:]
data[:,-1,:,:] = fn.variables[L[10]][:samples,:,:]
fn.close()
print('Swapping axis...')
data = np.swapaxes(data,1,2)
data = np.swapaxes(data,2,3)
print('done.')
data.astype(np.float32)
#np.swapaxes(data,3,4)
print('saving .h5 file.')
#np.save('../../data/raw/100yPlasim_81chans.h5', data)
print('data saved .npy file')
with h5.File('../../data/raw/100yPlasim_81chans.h5', 'w') as f:
    data = f.create_dataset("100yPlasim_81chans", (samples,81, 64, 128), chunks = True)
    '''