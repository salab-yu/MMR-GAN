import os
import numpy as np
from datetime import datetime
import tensorflow as tf
from config import FLAGS
from models import generator_model, discriminator_model
from data_loader import func_ge_A_data, func_ge_B_data
from utils import cal_loss, how_real
from random import shuffle

def main():
    A2B_gener = generator_model()
    B_dis = discriminator_model()
    generator_optimizer = tf.keras.optimizers.Adam(FLAGS.lr, beta_1=0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(FLAGS.lr, beta_1=0.5)
    
    if FLAGS.pre_checkpoint:
        ckpt = tf.train.Checkpoint(A2B_gener=A2B_gener, B_dis=B_dis,
                                   generator_optimizer=generator_optimizer,
                                   discriminator_optimizer=discriminator_optimizer)
        ckpt_manager = tf.train.CheckpointManager(ckpt, FLAGS.pre_checkpoint_path, 5)
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print("Checkpoint restored!")

    train_data = os.listdir(FLAGS.img_path)
    train_data = [os.path.join(FLAGS.img_path, data) for data in train_data]

    total_folder = os.path.join(FLAGS.save_path, f'Date_{datetime.today().strftime("%m%d")}')
    os.makedirs(total_folder, exist_ok=True)

    count = 0
    for epoch in range(FLAGS.epochs):
        shuffle(train_data)
        train_data = np.array(train_data)

        tr_idx = len(train_data) // FLAGS.batch_size
        for step in range(tr_idx):
            batch_images = train_data[step * FLAGS.batch_size:(step + 1) * FLAGS.batch_size]

            B_batch_images_buf = [func_ge_B_data(batch_images[i]) / 1.5 - 1. for i in range(FLAGS.batch_size)]
            A_batch_images_buf = [func_ge_A_data(B_batch_images_buf[i]) for i in range(FLAGS.batch_size)]

            B_batch_images_buf = np.expand_dims(np.array(B_batch_images_buf), -1)
            A_batch_images_buf = np.expand_dims(np.array(A_batch_images_buf), -1)

            G_loss, D_loss = cal_loss(A_batch_images_buf, B_batch_images_buf,
                                      A2B_gener, B_dis, generator_optimizer, discriminator_optimizer)

            if count % 20 == 0:
                print(f"Epochs: {epoch + 1} [{step}/{tr_idx}] G_loss = {G_loss:.3f}, D_loss = {D_loss:.3f}")

            if count % int(tr_idx / FLAGS.save_num) == 0:
                fake_B = A2B_gener(A_batch_images_buf, training=False)
                fake_img = (fake_B[:2] + 1) * 1.5
                fake_img = tf.squeeze(fake_img, -1)
                fake_img = np.round(fake_img)
                
                real_img = (B_batch_images_buf[:2] + 1) * 1.5
                real_img = tf.squeeze(real_img, -1)

                epoch_folder = os.path.join(total_folder, f'epoch_{epoch + 1}-{1 + int((count % tr_idx) // int(tr_idx / FLAGS.save_num))}')
                os.makedirs(epoch_folder, exist_ok=True)
                print(f"Make {epoch + 1}-{1 + int((count % tr_idx) // int(tr_idx / FLAGS.save_num))} folder to save samples")

                np.savetxt(os.path.join(epoch_folder, "fake-1.txt"), fake_img[0], fmt='%d', delimiter=',')
                np.savetxt(os.path.join(epoch_folder, "fake-2.txt"), fake_img[1], fmt='%d', delimiter=',')
                np.savetxt(os.path.join(epoch_folder, "real-1.txt"), real_img[0], fmt='%d', delimiter=',')
                np.savetxt(os.path.join(epoch_folder, "real-2.txt"), real_img[1], fmt='%d', delimiter=',')

                how_real(real_img, fake_img)
                        
                ckpt = tf.train.Checkpoint(A2B_gener=A2B_gener, B_dis=B_dis,
                                           generator_optimizer=generator_optimizer,
                                           discriminator_optimizer=discriminator_optimizer)
                ckpt.save(os.path.join(epoch_folder, f"GAN_{epoch}.ckpt"))

            count += 1