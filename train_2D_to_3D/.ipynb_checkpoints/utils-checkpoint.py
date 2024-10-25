import tensorflow as tf
from config import generator_optimizer, discriminator_optimizer
import numpy as np

def abs_criterion(input, target):
    return tf.reduce_mean(tf.abs(input - target))


def mae_criterion(input, target):
    return tf.reduce_mean((input - target)**2)


def cal_loss(A_batch_images_buf, B_batch_images_buf,
             A2B_gener, B_dis):

    with tf.GradientTape(persistent=True) as g_tape, tf.GradientTape(persistent=True) as d_tape:
        fake_B = A2B_gener(A_batch_images_buf, training=True)

        DB_fake = B_dis(fake_B, training=True)
        DB_real = B_dis(B_batch_images_buf, training=True)

        g_loss = mae_criterion(DB_fake, tf.ones_like(DB_fake)) + (10.0 * abs_criterion(B_batch_images_buf, fake_B))
        d_loss = (mae_criterion(DB_real, tf.ones_like(DB_real)) +
                       mae_criterion(DB_fake, tf.zeros_like(DB_fake))) / 2

    generator_gradients = g_tape.gradient(g_loss, A2B_gener.trainable_variables)
    discriminator_gradients = d_tape.gradient(d_loss, B_dis.trainable_variables)

    generator_optimizer.apply_gradients(zip(
        generator_gradients, A2B_gener.trainable_variables))

    discriminator_optimizer.apply_gradients(zip(
        discriminator_gradients, B_dis.trainable_variables))

    return g_loss, d_loss


def how_real(Real, Fake):
    Real = (Real + 1) * 1.5
    real_0 = len(np.where(Real==0)[0])
    real_1 = len(np.where(Real==1)[0])
    real_2 = len(np.where(Real==2)[0])
    real_3 = len(np.where(Real==3)[0])

    fake_0 = len(np.where(Fake==0)[0])
    fake_1 = len(np.where(Fake==1)[0])
    fake_2 = len(np.where(Fake==2)[0])
    fake_3 = len(np.where(Fake==3)[0])
    
    print("Num of 0 => Real: {} / Fake: {}".format(real_0, fake_0))
    print("Num of 1 => Real: {} / Fake: {}".format(real_1, fake_1))
    print("Num of 2 => Real: {} / Fake: {}".format(real_2, fake_2))
    print("Num of 3 => Real: {} / Fake: {}".format(real_3, fake_3))