import easydict
import tensorflow as tf

# Configuration settings using EasyDict
FLAGS = easydict.EasyDict({
    "img_path": "./full_phase",  # Path to the input image data
    "save_path2D": "./fake/2D_img",  # Path to save 2D generated images
    "save_path3D": "./fake/3D_img",  # Path to save 3D generated images
    "lr": 0.0002,  # Learning rate for the optimizers
    "batch_size": 1,  # Batch size for training
    "input_shape": (128, 128, 1),  # Shape of the input images for 2D models
    "input_shape_3D": (128, 128, 3),  # Shape of the input images for 3D models
    "int_filt_size": 32,  # Initial filter size for the 2D generator and discriminator models
    "int_filt_size_3D": 4,  # Initial filter size for the 3D generator and discriminator models
    "pre_checkpoint_path_2D": "./weights/2D",  # Path to load pre-trained 2D model checkpoints
    "pre_checkpoint_path_3D": "./weights/3D",  # Path to load pre-trained 3D model checkpoints
    # Additional parameters
    "random_noise": 3,  # Percentage of random noise to be applied to the fake images
    "time_limit": 60,  # Time limit (in minutes) for certain operations
    "how_many": 1,  # Number of times to perform a specific operation
    "dir_num": range(2),  # Range of directory indices to process
    # Initial tolerance values for specific metrics or thresholds
    "initial_pore_tor": 10.0,  # Initial tolerance for pore (0.03)
    "initial_outer_tor": 10.0,  # Initial tolerance for outer product (0.01)
    "initial_ll_tor": 10.0  # Initial tolerance for "ll" (likely a specific metric or layer (0.1)
})

# Optimizers for the generator and discriminator models
generator_optimizer = tf.keras.optimizers.Adam(FLAGS.lr, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(FLAGS.lr, beta_1=0.5)

# Column values used for indexing or selecting specific columns in data processing
col_val = [i for i in range(FLAGS.input_shape[0])]