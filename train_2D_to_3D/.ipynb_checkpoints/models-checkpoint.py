import tensorflow as tf
from config import FLAGS

def generator_model():

    h = inputs = tf.keras.Input(shape = FLAGS.input_shape)

    # Encoder
    ############################################################################
    h = tf.keras.layers.Conv2D(
        filters=FLAGS.int_filt_size*4, kernel_size=(4, 4), padding="same")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    h = tf.keras.layers.Conv2D(
        filters=FLAGS.int_filt_size*8, kernel_size=(4, 4), padding="same")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    block_1 = h
    block_1 = tf.keras.layers.Conv2D(
        filters=FLAGS.int_filt_size*32, kernel_size=(4, 4), padding="same")(block_1)
    block_1 = tf.keras.layers.BatchNormalization()(block_1)
    block_1 = tf.keras.layers.ReLU()(block_1)
    block_1 = tf.expand_dims(block_1, -1)
    block_1 = tf.keras.layers.Conv3D(
        filters=FLAGS.int_filt_size*2, kernel_size=(4, 4, 4), padding="same")(block_1)
    block_1 = tf.keras.layers.BatchNormalization()(block_1)
    block_1 = tf.keras.layers.ReLU()(block_1)

    h = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2)(h)

    h = tf.keras.layers.Conv2D(
        filters=FLAGS.int_filt_size*8, kernel_size=(4, 4), padding="same")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    h = tf.keras.layers.Conv2D(
        filters=FLAGS.int_filt_size*16, kernel_size=(4, 4), padding="same")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)


    block_2 = h
    block_2 = tf.keras.layers.Conv2D(
        filters=FLAGS.int_filt_size*16, kernel_size=(4, 4), padding="same")(block_2)
    block_2 = tf.keras.layers.BatchNormalization()(block_2)
    block_2 = tf.keras.layers.ReLU()(block_2)
    block_2 = tf.expand_dims(block_2, -1)
    block_2 = tf.keras.layers.Conv3D(
        filters=FLAGS.int_filt_size*4, kernel_size=(4, 4, 4), padding="same")(block_2)
    block_2 = tf.keras.layers.BatchNormalization()(block_2)
    block_2 = tf.keras.layers.ReLU()(block_2)

    h = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2)(h)

    h = tf.keras.layers.Conv2D(
        filters=FLAGS.int_filt_size*16, kernel_size=(4, 4), padding="same")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    h = tf.keras.layers.Conv2D(
        filters=FLAGS.int_filt_size*32, kernel_size=(4, 4), padding="same")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    block_3 = h
    block_3 = tf.keras.layers.Conv2D(
        filters=FLAGS.int_filt_size*8, kernel_size=(4, 4), padding="same")(block_3)
    block_3 = tf.keras.layers.BatchNormalization()(block_3)
    block_3 = tf.keras.layers.ReLU()(block_3)
    block_3 = tf.expand_dims(block_3, -1)
    block_3 = tf.keras.layers.Conv3D(
        filters=FLAGS.int_filt_size*8, kernel_size=(4, 4, 4), padding="same")(block_3)
    block_3 = tf.keras.layers.BatchNormalization()(block_3)
    block_3 = tf.keras.layers.ReLU()(block_3)

    h = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding="same")(h)
    
    h = tf.keras.layers.Conv2D(
        filters=FLAGS.int_filt_size*64, kernel_size=(4, 4), padding="same")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    h = tf.keras.layers.Flatten()(h)
    h = tf.keras.layers.Dense(4096)(h)
    h = tf.keras.layers.Dropout(0.5)(h)
    h = tf.keras.layers.Reshape([16, 16, 16])(h)
    h = tf.expand_dims(h, -1)
    
    # Decoder
    ###########################################################################################
    h = tf.keras.layers.Conv3DTranspose(filters=FLAGS.int_filt_size*2*4, kernel_size=(2, 2, 2), strides=2,
                                        padding="same")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h = tf.concat([block_3, h], -1)

    h = tf.keras.layers.Conv3DTranspose(filters=FLAGS.int_filt_size*2*2, kernel_size=(2, 2, 2), strides=2,
                                        padding='same')(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h = tf.concat([block_2, h], -1)

    h = tf.keras.layers.Conv3DTranspose(filters=FLAGS.int_filt_size*2, kernel_size=(2, 2, 2), strides=2,
                                        padding="same")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h = tf.concat([block_1, h], -1)

    h = tf.keras.layers.Conv3D(
        filters=FLAGS.int_filt_size*2, kernel_size=(4, 4, 4), padding="same")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    h = tf.keras.layers.Conv3D(
        filters=FLAGS.int_filt_size, kernel_size=(4, 4, 4), padding="same")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h = tf.keras.layers.Conv3D(filters=1, kernel_size=(1, 1, 1))(h)
    h = tf.nn.tanh(h)

    return tf.keras.Model(inputs=inputs, outputs=h)

def discriminator_model():
    
    dim = FLAGS.int_filt_size*8
    dim_ = dim
    # 0
    h = inputs = tf.keras.Input(shape = (128, 128, 128, 1))

    # 1
    h = tf.keras.layers.Conv3D(dim, 4, strides=2, padding='same')(h)
    h = tf.keras.layers.LeakyReLU(alpha=0.2)(h)

    for _ in range(2):
        dim = min(dim * 2, dim_ * 8)
        h = tf.keras.layers.Conv3D(
            dim, 4, strides=2, padding='same', use_bias=False)(h)
        h = tf.keras.layers.BatchNormalization()(h)
        h = tf.keras.layers.LeakyReLU(alpha=0.2)(h)

    # 2
    dim = min(dim * 2, dim_ * 8)
    h = tf.keras.layers.Conv3D(
        dim, 4, strides=1, padding='same', use_bias=False)(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.LeakyReLU(alpha=0.2)(h)

    # 3
    h = tf.keras.layers.Conv3D(1, 4, strides=1, padding='same')(h)

    return tf.keras.Model(inputs=inputs, outputs=h)