import easydict
import tensorflow as tf

FLAGS = easydict.EasyDict({
    "A_img_path": "/home/swhong/projects/GAN/Data_set/training_data/3D/Discrete_phase2/",
    "B_img_path": "/home/swhong/projects/GAN/Data_set/training_data/3D/Full_phase/",
    "lab_path": "./Discrete_label.txt",
    "save_path": "./save_3D",
    "batch_size": 16,
    "epochs": 400,
    "lr": 0.0004,
    "input_shape": (128, 128, 1),
    "save_num": 8,
    "int_filt_size": 4,
    "pre_checkpoint": False,
    "pre_checkpoint_path": ""
})

generator_optimizer = tf.keras.optimizers.Adam(FLAGS.lr, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(FLAGS.lr, beta_1=0.5)
col_val = [i for i in range(FLAGS.input_shape[0])]