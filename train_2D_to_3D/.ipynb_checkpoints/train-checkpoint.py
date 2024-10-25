import os
import numpy as np
from datetime import datetime
from config import FLAGS, generator_optimizer, discriminator_optimizer, col_val
from models import generator_model, discriminator_model
from data_loader import func_ge_A_data, func_ge_B_data
from utils import cal_loss, how_real
import tensorflow as tf
from random import shuffle

def main():

    A2B_gener_3D = generator_model()
    B_dis_3D = discriminator_model()
    
    if FLAGS.pre_checkpoint:
        ckpt = tf.train.Checkpoint(A2B_gener_3D=A2B_gener_3D, B_dis_3D=B_dis_3D,
                                   generator_optimizer=generator_optimizer,
                                   discriminator_optimizer=discriminator_optimizer)
        ckpt_manager = tf.train.CheckpointManager(ckpt,
                                                  FLAGS.pre_checkpoint_path, 5)
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print("Checkpoint restored")

    A_train_img = np.loadtxt(FLAGS.lab_path, dtype="<U100", skiprows=0, usecols=0)
    B_train_img = np.loadtxt(FLAGS.lab_path, dtype="<U100", skiprows=0, usecols=1)
    train_data = list(zip(A_train_img, B_train_img))
    
    total_folder = '%s/Date_%s' % (FLAGS.save_path, datetime.today().strftime("%m%d"))
    if not os.path.isdir(total_folder):
        os.makedirs(total_folder)

    count = 0
    for epoch in range(FLAGS.epochs):

        shuffle(train_data)
        
        A_train_img, B_train_img = zip(*train_data)
        
        A_train_data = [FLAGS.A_img_path + data for data in A_train_img]
        B_train_data = [FLAGS.B_img_path + data for data in B_train_img]
        

        tr_idx = len(A_train_data) // FLAGS.batch_size
        for step in range(tr_idx):
            A_batch_images = A_train_data[step *
                                      FLAGS.batch_size:(step + 1)*FLAGS.batch_size]
            B_batch_images = B_train_data[step *
                                      FLAGS.batch_size:(step + 1)*FLAGS.batch_size]

            A_batch_images_buf = [func_ge_A_data(
                A_batch_images[i]) / 1.5 - 1. for i in range(FLAGS.batch_size)]
            B_batch_images_buf = [func_ge_B_data(
                B_batch_images[i]) / 1.5 - 1. for i in range(FLAGS.batch_size)]

            A_batch_images_buf = np.expand_dims(
                np.array(A_batch_images_buf), -1)
            B_batch_images_buf = np.expand_dims(
                np.array(B_batch_images_buf), -1)

            G_loss, D_loss = cal_loss(A_batch_images_buf, B_batch_images_buf,
                                      A2B_gener_3D, B_dis_3D)

            if count % 20 == 0:
                print("Epochs: {0} [{1}/{2}] G_loss = {3:0.3f}, D_loss = {4:0.3f}".format(epoch,
                                                    step, tr_idx, G_loss, D_loss))
                
            if count % int(tr_idx / FLAGS.save_num) == 0:
                fake_B = A2B_gener_3D(A_batch_images_buf, training=False)  # [B, 128, 128, 128, 1]
                
                fake_img = fake_B[0]  # [128, 128, 128, 1]
                fake_img = (fake_img + 1) * 1.5
                fake_img = tf.squeeze(fake_img, -1)  # [128, 128, 128]
                fake_img = np.round(fake_img)
                
                real_img = B_batch_images_buf[0]
                real_img = (real_img + 1) * 1.5
                real_img = np.squeeze(real_img, -1)  # [128, 128, 128]
                
                epoch_folder = '%s/epoch_%03d-%d' % (total_folder, epoch + 1,
                        1 + int((count % tr_idx) // int(tr_idx / FLAGS.save_num)))
                if not os.path.isdir(epoch_folder):
                    os.makedirs(epoch_folder)
                print("Make {}-{} folder to save samples".format(epoch + 1,
                        1 + int((count % tr_idx) // int(tr_idx / FLAGS.save_num))))
                
                for j in range(128):
                    fake_ = fake_img[j]  # [128, 128]
                    np.savetxt(epoch_folder + "/" + "fake_{0:03d}.txt".format(j+1),
                               fake_, fmt='%d', delimiter=',')
                    
                for j in range(128):
                    real_ = real_img[j]  # [128, 128]
                    np.savetxt(epoch_folder + "/" + "real_{0:03d}.txt".format(j+1),
                               real_, fmt='%d', delimiter=',')
                    
                how_real(B_batch_images_buf[0], fake_img)
                
                ckpt = tf.train.Checkpoint(A2B_gener_3D=A2B_gener_3D, B_dis_3D=B_dis_3D,
                                           generator_optimizer=generator_optimizer,
                                           discriminator_optimizer=discriminator_optimizer)
                model_dir = epoch_folder + "/" + "3D_GAN_{0}.ckpt".format(epoch)
                ckpt.save(model_dir)

            count += 1

if __name__ == "__main__":
    main()