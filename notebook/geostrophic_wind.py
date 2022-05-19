# Import of necessary packages and memory allocation.
import sys
sys.path.append("../src/modeling")
sys.path.append("../src/preprocessing")
sys.path.append("../src/preparation")

from SpectralNormalizationKeras import *
from data_preproc import *
from custom_classes import *
import keras
import tensorflow as tf
from keras import backend as K
from keras.backend.tensorflow_backend import set_session
import metpy.calc as mpcalc
import xarray as xr
from metpy.units import units
from pint import UnitRegistry
from pyproj import CRS
import cartopy
import cartopy.crs as ccrs
import cartopy.util
import h5py as h5
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spharm
from keras.models import load_model
from sklearn.decomposition import PCA




config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
config.gpu_options.allow_growth = True
# config.gpu_options.visible_device_list = str(1)
set_session(tf.Session(config=config))

field = np.random.normal(0, 1, (64, 128))
x = spharm.Spharmt(
    128,
    64,
    rsphere=6.4e6,
    gridtype="gaussian",
    legfunc="computed")
spec = x.grdtospec(field)
field_2 = x.spectogrd(spec)

fig, ax = plt.subplots(2, 1, figsize=(10, 10))
ax[0].imshow(field)
ax[1].imshow(field_2)


plt.figure(figsize=(10, 10))
plt.imshow(field_2)

# Loading Data

#########
multi_train = True  # If the training was in multiple run : True
plot_var = True  # True: plot all cells result ; False : manualy change if plot_var in if True: to plot
save = False
scale = True
#########
N_fields = 82  # Number of fields == Number of channels in generated tensor.
N_lvls = 10  # Number of pressure level for 3D variable.

years = 1
N_train = 360 * years
# Number of generation used for statistic analysis
# /!\ Memory usage important /!\
N_gen = 360 * years
scaled = True
#########
lon = np.genfromtxt("../data/raw/lon.csv", delimiter=",")

lons = np.genfromtxt("../data/raw/lon.csv", delimiter=",")
lats = np.genfromtxt("../data/raw/lat.csv", delimiter=",")

print(lons)
lon_idx = [16, 16, 32, 48]  # France, north america, ecuador, south america
lat_idx = [5, 106, 106, 106]  # France, north america, ecuador, south america
fnes = [
    [16, 5],
    [16, 106],
    [32, 106],
    [48, 106],
]  # Coordinate for distribution comparison at different locations.
L_ind = ["a", "b", "c", "d"]

runnumber_0 = "RESNET_V8_82l"
runnumber_1, stop1 = (
    "RESNET_V8_82l_continue1",
    8000,
)  # Name of the generator save file for the 1st restart
runnumber_2, stop2 = (
    "RESNET_V8_82l_continue2",
    15500,
)  # Name of the generator save file for the 2nd restart
runnumber_3, stop3 = "RESNET_V8_82l_continue3b", 23250
runnumber_4, stop4 = "RESNET_V8_82l_continue4", 31000
runnumber = runnumber_4  # Most recent save


try:
    gen = load_model(
        f"../model/{runnumber}/{runnumber}_generator.h5",
        custom_objects={
            "NearestPadding2D": NearestPadding2D,
            "WrapPadding2D": WrapPadding2D,
            "DenseSN": DenseSN,
            "ConvSN2D": ConvSN2D,
        },
    )
except Exception:
    gen = load_model(
        f"../model/{runnumber}_generator.h5",
        custom_objects={
            "NearestPadding2D": NearestPadding2D,
            "WrapPadding2D": WrapPadding2D,
            "DenseSN": DenseSN,
            "ConvSN2D": ConvSN2D,
        },
    )
gen._make_predict_function()

print("Database loading...")

f = h5.File("../data/raw/data_plasim_3y_sc.h5", "r")
_X_train = f["dataset"]
scaling = f["scaling"][:, :]

print("Database loaded.")
print("Scaling...")
if scaled:
    X_train = np.multiply(_X_train[:N_train, :, :, :],
                          scaling[np.newaxis, :, 1]) + scaling[np.newaxis, :, 0]
else:
    X_train = _X_train[:N_train]
print("Scaled")

print("Generating samples...")

z = np.random.normal(0, 1, (N_gen, 64))
fk_imgs = gen.predict(z)
print("Samples generated.")
print("scaling...")
if scaled:
    fk_imgs = (
        np.multiply(fk_imgs[:, :, :, :-1], scaling[np.newaxis, :, 1])
        + scaling[np.newaxis, :, 0]
    )
print("scaled.")


noise_ = np.load("../data/raw/noise_.npy")
im = gen.predict(noise_)
image = (np.multiply(im[:, :, :, :-
                        1], scaling[np.newaxis, :, 1]) +
         scaling[np.newaxis, :, 0])
image.shape

ind = 10

X_train = (
    np.multiply(_X_train[ind:ind + 1, :, :, :], scaling[np.newaxis, :, 1])
    + scaling[np.newaxis, :, 0])
print(X_train.shape)
geopot, lons = cartopy.util.add_cyclic_point(X_train[:, :, :, 74],
                                             coord=lons, axis=2)
print(geopot.shape, len(lons))
phi, theta = np.meshgrid(lons, lats)
print(theta, phi.shape)

plt.colorbar()

ureg = UnitRegistry()
lats_p = lats * ureg.deg
lons_p = lons * ureg.deg
phi, theta = np.meshgrid(lons, lats) * ureg.deg
f = mpcalc.coriolis_parameter(theta)
dx, dy = mpcalc.lat_lon_grid_deltas(phi, theta) * ureg.meter

print(f.shape)
print(dx.shape)

heights = geopot * ureg.hPa
heights = xr.DataArray(
    geopot[0, :, :] * units["hPa"],
    dims=("lat", "lon"),
    coords={"lon": lons_p, "lat": lats_p},
)
print(heights)

# heights = geopot.metpy.loc[{'vertical': 500. * units.hPa}]
u_geo, v_geo = mpcalc.geostrophic_wind(height=heights, dx=dx, dy=dy, latitude=lats_p)

print(u_geo)
print(u_geo.m)

plt.quiver(phi, theta, u_geo, v_geo)

# Or, let's make a full 500 hPa map with heights, temperature, winds, and
# humidity

# Select the data for this time and level
# data_level = data.metpy.loc[{time.name: time[0], vertical.name: 500. * units.hPa}]
extent = [-60, 60, 35, 90]
# Create the matplotlib figure and axis
fig, ax = plt.subplots(
    1,
    1,
    figsize=(12, 8),
    subplot_kw={"projection": ccrs.PlateCarree(central_longitude=0.0)},
)
ax.set_extent(extent, crs=ccrs.PlateCarree())

# Plot wind barbs, but not all of them
wind_slice = slice(5, -5, 5)
ax.barbs(
    theta[wind_slice],
    phi[wind_slice],
    u_geo[wind_slice],
    v_geo[wind_slice])

plt.show()
