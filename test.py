def training_test():
    # ############IMPORT##########################
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
    from src.modeling import wgan_gp_V8_82c as wg
    from src.preparation import data_preproc as preproc

    # Setting for memory allocaton of the GPU.
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))
    sess = tf.Session()

    X_train, scaling = preproc.dataExtraction_puma(
        DB_path='./data/raw/data_plasim_3y_sc.h5',
        DB_name='dataset', im_shape=(64, 128, 81)
        )

    wgan = wg.WGANGP(latent_dim=64, target_shape=(64, 128, 82), batch_size=2,
                     optimizerG=None, optimizerC=None, summary=True,
                     n_critic=1, models=None, gradient_penalty=10,
                     data=X_train[:100, :, :, :], tfboard=False)

    wgan.train(epochs=5, save_interval=4, save_file='test',
               run_name='run_test', log_interval=10, log_file='test',
               data_generator=None, save_intermediate_model=True)
