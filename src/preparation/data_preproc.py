import numpy as np
import h5py as h5
import copy as cp


def dataExtraction_puma(DB_path="", DB_name="", im_shape=(64, 128, 5)):
    """
    Returns a numpy array with data from .h5 file.
    """
    rows = im_shape[0]
    cols = im_shape[1]
    chans = im_shape[2]
    scaling = np.zeros((chans, 2))  # contain mean and var for the 4 chans
    DB_size = 3 * 365
    print(f"Size of the DB : {DB_size}")
    DB_images = np.ones((DB_size, rows, cols, chans))

    f = h5.File(DB_path, "r")
    f[DB_name].read_direct(DB_images)

    print("Dataset used : 1095 images 81 channels - 3y simulation")
    for chan in range(chans):
        DB_images[:, :, :, chan], scaling[chan, 0], scaling[chan, 1] = scale2(
            DB_images[:, :, :, chan]
        )
        inter = DB_images[:, :, :, chan]
    f.close()

    return DB_images, scaling


def dataPreparationPuma(DB_path="", DB_name="", im_shape=(64, 128, 81)):
    """
    Returns a .h5 file with standardized data and the means and stds used for
    the standardization.
    """
    rows = im_shape[0]
    cols = im_shape[1]
    chans = im_shape[2]
    scaling = np.zeros((chans, 2))  # contain mean and var for the 4 chans
    DB_size = 3 * 365
    print("Size of the DB : ", DB_size)
    DB_images = np.ones((DB_size, rows, cols, chans))

    f = h5.File(DB_path, "r")

    f[DB_name].read_direct(DB_images)
    print("Dataset used : k images 10 channels - 50y simulation")
    for chan in range(chans):
        DB_images[:, :, :, chan], scaling[chan, 0], scaling[chan, 1] = scale2(
            DB_images[:, :, :, chan]
        )
        inter = DB_images[:, :, :, chan]
        # scaling[chan,0] = np.mean(inter,axis = 0)
        # scaling[chan,1] = np.std(inter,axis = 0)**2
    x_trainf = cp.copy(DB_images)
    print("saving...")
    with h5.File("../../data/raw/data_plasim_3y_sc.h5", "w") as f:
        dset = f.create_dataset("dataset", data=x_trainf)
        dset = f.create_dataset("scaling", data=scaling)
        print(list(f.keys()))
        # dset = f['mydataset']
        # print(dset.shape)
    print("saving done")
    f.close()

    return


def scale2(X):
    mean = np.mean(X)
    X = X[:] - mean
    m = np.std(X)
    X1 = X[:] / m

    return X1, mean, m
