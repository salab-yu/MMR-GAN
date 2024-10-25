import tensorflow as tf
from config import FLAGS

def generator_model():
    h = inputs = tf.keras.Input(FLAGS.input_shape)
    # Encoder
    h = tf.keras.layers.Conv2D(filters=FLAGS.int_filt_size, kernel_size=(4, 4), padding="same")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    h = tf.keras.layers.Conv2D(filters=FLAGS.int_filt_size * 2, kernel_size=(4, 4), padding="same")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    block_1 = h
    h = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2)(h)
    h = tf.keras.layers.Conv2D(filters=FLAGS.int_filt_size * 2, kernel_size=(4, 4), padding="same")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    h = tf.keras.layers.Conv2D(filters=FLAGS.int_filt_size * 4, kernel_size=(4, 4), padding="same")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    block_2 = h
    h = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2)(h)
    h = tf.keras.layers.Conv2D(filters=FLAGS.int_filt_size * 4, kernel_size=(4, 4), padding="same")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    h = tf.keras.layers.Conv2D(filters=FLAGS.int_filt_size * 8, kernel_size=(4, 4), padding="same")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    block_3 = h
    h = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding="same")(h)
    h = tf.keras.layers.Conv2D(filters=FLAGS.int_filt_size * 8, kernel_size=(4, 4), padding="same")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    h = tf.keras.layers.Conv2D(filters=FLAGS.int_filt_size * 16, kernel_size=(4, 4), padding="same")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    block_4 = h
    # Decoder
    h = tf.keras.layers.Conv2DTranspose(filters=FLAGS.int_filt_size * 8, kernel_size=(2, 2), strides=2, padding="same")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    h = tf.concat([block_3, h], -1)
    h = tf.keras.layers.Conv2DTranspose(filters=FLAGS.int_filt_size * 4, kernel_size=(2, 2), strides=2, padding='same')(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    h = tf.concat([block_2, h], -1)
    h = tf.keras.layers.Conv2DTranspose(filters=FLAGS.int_filt_size * 2, kernel_size=(2, 2), strides=2, padding="same")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    h = tf.concat([block_1, h], -1)
    h = tf.keras.layers.Conv2D(filters=FLAGS.int_filt_size * 2, kernel_size=(4, 4), padding="same")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    h = tf.keras.layers.Conv2D(filters=FLAGS.int_filt_size, kernel_size=(4, 4), padding="same")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    h = tf.keras.layers.Conv2D(filters=1, kernel_size=(1, 1))(h)
    h = tf.nn.tanh(h)
    return tf.keras.Model(inputs=inputs, outputs=h)

def discriminator_model():
    dim = FLAGS.int_filt_size * 2
    dim_ = dim
    h = inputs = tf.keras.Input(FLAGS.input_shape)
    h = tf.keras.layers.Conv2D(dim, 4, strides=2, padding='same')(h)
    h = tf.keras.layers.LeakyReLU(alpha=0.2)(h)
    for _ in range(2):
        dim = min(dim * 2, dim_ * 8)
        h = tf.keras.layers.Conv2D(dim, 4, strides=2, padding='same', use_bias=False)(h)
        h = tf.keras.layers.BatchNormalization()(h)
        h = tf.keras.layers.LeakyReLU(alpha=0.2)(h)
    dim = min(dim * 2, dim_ * 8)
    h = tf.keras.layers.Conv2D(dim, 4, strides=1, padding='same', use_bias=False)(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.LeakyReLU(alpha=0.2)(h)
    h = tf.keras.layers.Conv2D(1, 4, strides=1, padding='same')(h)
    return tf.keras.Model(inputs=inputs, outputs=h)