import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

def abs_criterion(input, target):
    return tf.reduce_mean(tf.abs(input - target))

def mae_criterion(input, target):
    return tf.reduce_mean((input - target) ** 2)

def cal_loss(A_batch_images_buf, B_batch_images_buf, A2B_gener, B_dis, generator_optimizer, discriminator_optimizer):
    with tf.GradientTape(persistent=True) as g_tape, tf.GradientTape(persistent=True) as d_tape:
        fake_B = A2B_gener(A_batch_images_buf, training=True)
        DB_fake = B_dis(fake_B, training=True)
        DB_real = B_dis(B_batch_images_buf, training=True)
        g_loss = mae_criterion(DB_fake, tf.ones_like(DB_fake)) + (10.0 * abs_criterion(B_batch_images_buf, fake_B))
        d_loss = (mae_criterion(DB_real, tf.ones_like(DB_real)) +
                  mae_criterion(DB_fake, tf.zeros_like(DB_fake))) / 2
    generator_gradients = g_tape.gradient(g_loss, A2B_gener.trainable_variables)
    discriminator_gradients = d_tape.gradient(d_loss, B_dis.trainable_variables)
    generator_optimizer.apply_gradients(zip(generator_gradients, A2B_gener.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients, B_dis.trainable_variables))
    return g_loss, d_loss

def how_real(Real, Fake):
    Real_img = np.clip(Real, 0, 255).astype('uint8')
    Fake_img = np.clip(Fake, 0, 255).astype('uint8')
    figure, ax = plt.subplots(nrows=2, ncols=2, figsize=(5, 5))
    ax.ravel()[0].imshow(Fake_img[0], cmap='gray')
    ax.ravel()[0].set_title('Fake img 1', size=15, color='white')
    ax.ravel()[0].axis('off')
    ax.ravel()[1].imshow(Real_img[0], cmap='gray')
    ax.ravel()[1].set_title('Real img 1', size=15, color='white')
    ax.ravel()[1].axis('off')
    ax.ravel()[2].imshow(Fake_img[1], cmap='gray')
    ax.ravel()[2].set_title('Fake img 2', size=15, color='white')
    ax.ravel()[2].axis('off')
    ax.ravel()[3].imshow(Real_img[1], cmap='gray')
    ax.ravel()[3].set_title('Real img 2', size=15, color='white')
    ax.ravel()[3].axis('off')
    plt.tight_layout()
    plt.show()