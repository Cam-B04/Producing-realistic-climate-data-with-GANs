
import h5py as h5 
import h5netcdf
import netCDF4
import numpy as np



#with h5netcdf.File('./mydata100yplasim.nc', 'r') as f:

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

chans = 10
fn = netCDF4.Dataset('../../data/raw/100y_10layers.nc', 'r')
L = list(fn.variables.keys())
print(fn.variables.keys())
print(fn)
#print(fn['ta'])
for i in L:
    print(fn.variables[i])


data = np.zeros((30000,81, 64, 128))
for i in range(4,12):
    if i==10:
    	i=11
    	data[:,6*chans:7*chans,:,:] = fn.variables[L[12]][:30000,:chans,:,:]
        #data[:,(i-4)*chans,:,:] = fn.variables[L[i]][:30000,:,:]
    else:
        data[:,(i-4)*chans:(i+1-4)*chans,:,:] = fn.variables[L[i]][:30000,:chans,:,:]
    print('Variables ',i,'/5... ', L[i])
#data[:,(i-4)*chans:(i+1-4)*chans,:,:] = fn.variables[L[i]][:30000,:chans,:,:]
data[:,-1,:,:] = fn.variables[L[10]][:30000,:,:]
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
    data = f.create_dataset("100yPlasim_81chans", (30000,81, 64, 128), chunks = True)
    