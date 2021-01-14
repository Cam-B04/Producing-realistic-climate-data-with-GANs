#Import of necessary packages and memory allocation.


import tensorflow as tf
import keras 
from keras import backend as K
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
config.gpu_options.allow_growth = True
#config.gpu_options.visible_device_list = str(1)
set_session(tf.Session(config=config))
import time
import h5py as h5
import pandas as pd
from keras.models import load_model
import numpy as np 
import matplotlib
import matplotlib.pyplot as plt

import sys
import seaborn as sns
import math
from sklearn.decomposition import PCA
#s = np.random.seed(1)




import cartopy
import cartopy.crs as ccrs
import cartopy.util
import spharm
import numpy as np



import cartopy
import cartopy.crs as ccrs
import cartopy.util


import spharm
import numpy as np

field = np.random.normal(0,1, (64, 128))
x = spharm.Spharmt(128, 64, rsphere=6.4e6, gridtype='gaussian', legfunc='computed')

spec = x.grdtospec(field)

field_2 = x.spectogrd(spec)

#plt.figure(figsize=(10,10))
fig, ax = plt.subplots(2,1, figsize=(10,10))
ax[0].imshow(field)
ax[1].imshow(field_2)
#plt.colorbar()

plt.figure(figsize=(10,10))
plt.imshow(field_2)
#plt.colorbar()



## Loading Data

#########
multi_train = True  #If the training was in multiple run : True
plot_var = True     #True: plot all cells result ; False : manualy change if plot_var in if True: to plot
save = False         
scale = True
#########
N_fields = 82      #Number of fields == Number of channels in generated tensor. 
N_lvls = 10         #Number of pressure level for 3D variable. 

years = 14
N_train = 360*years
N_gen = 1*years     #Number of generation used for statistic analysis /!\ Memory usage important /!\
scaled = True

#indxs = np.random.randint(0,1000,N_gen)
#ind = indxs[0]
#########
lon = np.genfromtxt('../data/raw/lon.csv', delimiter=',')

lons = np.genfromtxt('../data/raw/lon.csv', delimiter=',')
lats = np.genfromtxt('../data/raw/lat.csv', delimiter=',')

print(lons)
lon_idx = [16,16,32,48]                    #France, north america, ecuador, south america
lat_idx = [5,106,106,106]                  #France, north america, ecuador, south america
fnes = [[16,5],[16,106],[32,106],[48,106]] #Coordinate for distribution comparison at different locations.
L_ind = ['a','b','c','d']


if N_fields==82:
    runnumber_0 = 'RESNET_V8_82l'
    runnumber_1, stop1 = 'RESNET_V8_82l_continue1', 8000 #Name of the generator save file for the 1st restart
    runnumber_2, stop2 = 'RESNET_V8_82l_continue2', 15500 #Name of the generator save file for the 2nd restart
    runnumber_3, stop3 = 'RESNET_V8_82l_continue3b', 23250
    runnumber_4, stop4 = 'RESNET_V8_82l_continue4', 31000
    runnumber = runnumber_4 #Most recent save
elif N_fields==22:
    runnumber_0 = 'RESNET_V8_gual_Gbn'
    runnumber = runnumber_0 #Most recent save



sys.path.append('../src/modeling')

sys.path.append('../src/preprocessing')

sys.path.append('../src/preparation')
from SpectralNormalizationKeras import *
from custom_classes import *
from data_preproc import *

try:
    gen = load_model(f'../model/save/{runnumber}/{runnumber}_generator.h5',custom_objects = {'NearestPadding2D': NearestPadding2D,'WrapPadding2D': WrapPadding2D, 'DenseSN' :DenseSN, 'ConvSN2D': ConvSN2D})
except:
    gen = load_model(f'../model/{runnumber}_generator.h5',
          custom_objects = {'NearestPadding2D': NearestPadding2D,'WrapPadding2D': WrapPadding2D, 'DenseSN' :DenseSN, 'ConvSN2D': ConvSN2D})
gen._make_predict_function()

print('Database loading...')

if N_fields==82:
    f=h5.File('/Users/besombes/Work/python/Cerfacs/Dataset/T42_plasim_100y_10lay_scaled.h5','r')
    _X_train = f['dataset']
    scaling = np.transpose(f['scaling'])
elif N_fields==22:
    X_train = np.load('../data/raw/x_train_22c_scaled.npy')
    scaling = np.load('../data/raw/scaling_21c.npy')
    
print('Database loaded.')
print('Scaling...')
#if scaled:
#    X_train = np.multiply(_X_train[:N_train,:,:,:],
#                          scaling[np.newaxis,:,1]) + scaling[np.newaxis,:,0]
#else:
#    X_train = _X_train[:N_train]
print('Scaled')

print('Generating samples...')

z = np.random.normal(0,1,(N_gen,64))
fk_imgs = gen.predict(z)
print('Samples generated.')
print('scaling...')
if scaled:
    fk_imgs = np.multiply(fk_imgs[:,:,:,:-1],
                          scaling[np.newaxis,:,1]) + scaling[np.newaxis,:,0]
print('scaled.')


#noise_ = np.load('./fig_82c/noise_.npy')
#im = gen.predict(noise_)
#im_sc = np.multiply(im[:,:,:,:-1],
#                    scaling[np.newaxis,:,1]) + scaling[np.newaxis,:,0]




noise_ = np.load('../data/raw/noise_.npy')
im = gen.predict(noise_)
image = np.multiply(im[:,:,:,:-1],
                          scaling[np.newaxis,:,1]) + scaling[np.newaxis,:,0]
image.shape
#im_sc = np.multiply(im[:,:,:,:-1],
#                    scaling[np.newaxis,:,1]) + scaling[np.newaxis,:,0]

ind = 4115
#print(lons)
#lons = np.genfromtxt('/scratch/coop/besombes/Puma_Project/data/raw/lon.csv', delimiter=',')
#lons_n = np.genfromtxt('/scratch/coop/besombes/Puma_Project/data/raw/lon.csv', delimiter=',')
#lats = np.genfromtxt('/scratch/coop/besombes/Puma_Project/data/raw/lat.csv', delimiter=',')

X_train = np.multiply(_X_train[ind:ind+1,:,:,:],
                      scaling[np.newaxis,:,1]) + scaling[np.newaxis,:,0]
print(X_train[0,:,:,74].shape)
geopot, lons = cartopy.util.add_cyclic_point(X_train[:,:,:,74], coord=lons, axis=2)
print(geopot.shape, len(lons))
phi, theta = np.meshgrid(lons,lats)
print(theta, phi.shape)

plt.imshow(geopot[0,:,:])
plt.colorbar()

from metpy.units import units
import metpy as mt
from pint import UnitRegistry
import metpy.calc as mpcalc
import xarray as xr
from pyproj import CRS


ureg = UnitRegistry()
lats_p = lats*ureg.deg
lons_p = lons*ureg.deg


#lats_p.dims = units.deg
#lons_p.dims = units.deg

phi, theta = np.meshgrid(lons,lats)*ureg.deg

#f = 2.*7.2921e-5*np.sin(theta*np.pi/180.)/ureg.second




#dy = dy*units.meters
#dx = dx*units.meters

#print(dx.dim, dy)
f = mpcalc.coriolis_parameter(theta)
#f, lons = cartopy.util.add_cyclic_point(f, coord=lon, axis=1)

dx, dy = mpcalc.lat_lon_grid_deltas(phi, theta)

print(f.shape)
#f = np.repeat(f, 129, axis=1 )
print(dx.shape)

#lat, lon = xr.broadcast(lats_p, lons_p)
heights = geopot*ureg.hPa
heights = xr.DataArray(geopot[0,:,:]*units['hPa'], dims=("lat", "lon"), coords={"lon": lons_p, "lat" : lats_p})
print(heights)



#heights = geopot.metpy.loc[{'vertical': 500. * units.hPa}]
u_geo, v_geo = mpcalc.geostrophic_wind(heights, f, dx, dy)

print(u_geo)
print(u_geo.m)

plt.quiver(phi, theta, u_geo, v_geo)

#extent=[-60, 60, 35, 90]
#fig, axc = plt.subplots(nrows = 2, ncols = 1, figsize=(15,20),
#            subplot_kw={'projection': ccrs.PlateCarree(central_longitude = 0.)})
##plt.subplots_adjust(top=0.9, bottom=0.001,right=0.89, left=0.1, wspace=0.1, hspace=0.1)
#axc[0].set_extent(extent, crs=ccrs.PlateCarree())
##axc[0].coastlines(linewidth = 3.)
#pgeo = axc[0].quiver(phi, theta, 
#                     u_geo, v_geo)
##                     transform=ccrs.PlateCarree(central_longitude=360.))

# Or, let's make a full 500 hPa map with heights, temperature, winds, and humidity
import cartopy.feature as cfeature
# Select the data for this time and level
#data_level = data.metpy.loc[{time.name: time[0], vertical.name: 500. * units.hPa}]
extent=[-60, 60, 35, 90]
# Create the matplotlib figure and axis
fig, ax = plt.subplots(1, 1, figsize=(12, 8), subplot_kw={'projection': ccrs.PlateCarree(central_longitude = 0.)})
ax.set_extent(extent, crs=ccrs.PlateCarree())
# Plot RH as filled contours
#rh = ax.contourf(phi, theta, geopot[0,:,:],
#                 colors=['#99ff00', '#00ff00', '#00cc00'])

# Plot wind barbs, but not all of them
wind_slice = slice(5, -5, 5)
ax.barbs(theta[wind_slice], phi[wind_slice],
         u_geo[wind_slice],
         v_geo[wind_slice])

# Plot heights and temperature as contours
#h_contour = ax.contour(x, y, data_level['height'], colors='k', levels=range(5400, 6000, 60))
#h_contour.clabel(fontsize=8, colors='k', inline=1, inline_spacing=8,
#                 fmt='%i', rightside_up=True, use_clabeltext=True)
#t_contour = ax.contour(x, y, data_level['temperature'], colors='xkcd:deep blue',
#                       levels=range(-26, 4, 2), alpha=0.8, linestyles='--')
#t_contour.clabel(fontsize=8, colors='xkcd:deep blue', inline=1, inline_spacing=8,
#                 fmt='%i', rightside_up=True, use_clabeltext=True)
#
# Add geographic features
#ax.coastlines()

# Set a title and show the plot
plt.show()