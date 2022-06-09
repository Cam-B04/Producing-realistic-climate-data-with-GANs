def loading_database():
    '''Checks if the database loading steps works'''
    from src.preparation import data_preproc as preproc
    import os

    if os.path.exists('./data/raw/data_plasim_3y_sc.h5'):
        X_train, scaling = preproc.dataExtraction_puma(
            DB_path='./data/raw/data_plasim_3y_sc.h5',
            DB_name='dataset', im_shape=(64, 128, 81)
            )
    return


def init_wgan():
    '''Checks if the WGAN initialization steps works with fake database'''
    import numpy as np
    import sys
    sys.path.append('./src/modeling')
    import wgan_gp_V8_82c as wg

    X_train = np.zeros((2, 64, 128, 81))
    wgan = wg.WGANGP(latent_dim=64, target_shape=(64, 128, 82), batch_size=2,
                     optimizerG=None, optimizerC=None, summary=True,
                     n_critic=1, models=None, gradient_penalty=10,
                     data=X_train, tfboard=False)
    return


def training_wgan():
    '''Checks if the WGAN training step works. 2 iterations on fake data.'''
    import numpy as np
    import sys
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
    sys.path.append('./src/modeling')
    import wgan_gp_V8_82c as wg

    # Setting for memory allocaton of the GPU.
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))
    sess = tf.Session()

    X_train = np.zeros((2, 64, 128, 81))
    wgan = wg.WGANGP(latent_dim=64, target_shape=(64, 128, 82), batch_size=2,
                     optimizerG=None, optimizerC=None, summary=True,
                     n_critic=1, models=None, gradient_penalty=10,
                     data=X_train, tfboard=False)

    wgan.train(epochs=2, save_interval=1, save_file='test',
               run_name='run_test', log_interval=10, log_file='test',
               data_generator=None, save_intermediate_model=True)
    return


def test_loading_database():
    assert loading_database() is None


def test_init_wgan():
    assert init_wgan() is None


def test_training_wgan():
    assert training_wgan() is None
