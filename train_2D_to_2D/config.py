import easydict
import tensorflow as tf

FLAGS = easydict.EasyDict({
    "img_path": "/home/swhong/projects/MMR_GAN/full_phase/original_phase_00010",
    "save_path": "./save_2D",
    "batch_size": 128,
    "epochs": 100,
    "lr": 0.00001,
    "input_shape": (128, 128, 1),
    "int_filt_size": 32,
    "save_num": 8,
    "pre_checkpoint": False,
    "pre_checkpoint_path": "-"
})

generator_optimizer = tf.keras.optimizers.Adam(FLAGS.lr, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(FLAGS.lr, beta_1=0.5)
col_val = [i for i in range(FLAGS.input_shape[0])]