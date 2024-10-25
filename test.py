import os
import numpy as np
import tensorflow as tf
from datetime import datetime
from config import FLAGS, generator_optimizer, discriminator_optimizer, col_val
from models import generator_model_2D, discriminator_model_2D, generator_model_3D, discriminator_model_3D
from data_loader import func_ge_A_data_3D
from utils import ge_noise, img_smooth, l2_rec1_3D

def main():
    A2B_gener_2D = generator_model_2D()
    B_dis_2D = discriminator_model_2D()
    A2B_gener_3D = generator_model_3D()
    B_dis_3D = discriminator_model_3D()
    
    ckpt_2D = tf.train.Checkpoint(A2B_gener=A2B_gener_2D, B_dis=B_dis_2D,
                               generator_optimizer=generator_optimizer,
                               discriminator_optimizer=discriminator_optimizer)
    ckpt_manager_2D = tf.train.CheckpointManager(ckpt_2D,
                                              FLAGS.pre_checkpoint_path_2D, 5)
    if ckpt_manager_2D.latest_checkpoint:
        ckpt_2D.restore(ckpt_manager_2D.latest_checkpoint)
        print("2D parameter restored")

    ckpt_3D = tf.train.Checkpoint(A2B_gener_3D=A2B_gener_3D, B_dis_3D=B_dis_3D,
                               generator_optimizer=generator_optimizer,
                               discriminator_optimizer=discriminator_optimizer)
    ckpt_manager_3D = tf.train.CheckpointManager(ckpt_3D,
                                              FLAGS.pre_checkpoint_path_3D, 5)
    if ckpt_manager_3D.latest_checkpoint:
        ckpt_3D.restore(ckpt_manager_3D.latest_checkpoint)
        print("3D parameter restored")

    test_data = os.listdir(FLAGS.img_path)
    test_data = [FLAGS.img_path + "/" + data for data in test_data]
    test_data = np.array(test_data)
    test_data = sorted(test_data)

    for dir_n in FLAGS.dir_num:
        real_img = func_ge_A_data_3D(test_data[dir_n])
        real_img = np.array(real_img)
        rmap_0 = np.where(real_img == 0)
        rmap_1 = np.where(real_img == 1)
        rmap_2 = np.where(real_img == 2)
        rmap_3 = np.where(real_img == 3)
        r_phi = len(rmap_0[0])/(len(real_img)*len(real_img)*len(real_img))
        r_out = len(rmap_1[0])/(len(real_img)*len(real_img)*len(real_img))
        r_in = len(rmap_2[0])/(len(real_img)*len(real_img)*len(real_img))
        r_unh = len(rmap_3[0])/(len(real_img)*len(real_img)*len(real_img))
        
        # calculate for pore
        real_phi = real_img * 0
        real_phi[rmap_0] = 1
        r_phi_l2rec = l2_rec1_3D(real_phi)
        r_phi_xa = np.array(r_phi_l2rec[:27,1])
        r_phi_ya = np.array(r_phi_l2rec[:27,2])
        r_phi_za = np.array(r_phi_l2rec[:27,3])
        r_phi_l2 = (r_phi_xa + r_phi_ya + r_phi_za) / 3

        # calculate for outer product
        real_out = real_img * 0
        real_out[rmap_1] = 1
        r_out_l2rec = l2_rec1_3D(real_out)
        r_out_xa = np.array(r_out_l2rec[:27,1])
        r_out_ya = np.array(r_out_l2rec[:27,2])
        r_out_za = np.array(r_out_l2rec[:27,3])
        r_out_l2 = (r_out_xa + r_out_ya + r_out_za) / 3


        test_3D_input = np.zeros((128,128,3))

        time_out = True
        counts1 = 0
        counts2 = 0
        pore_tor = FLAGS.initial_pore_tor
        outer_tor = FLAGS.initial_outer_tor
        ll_tor = FLAGS.initial_ll_tor
        
####### 2D reconstruction
        time1 = datetime.now()
        phi_counts = 0
        out_counts = 0
        ll_counts = 0
        while counts2 < FLAGS.how_many and time_out:
            fake_imgs = np.zeros(FLAGS.input_shape_3D)
            for img_num in range(3):
                temp_num_pixel = [r_phi, r_out, r_in, r_unh]
                temp_num_pixel = np.array(temp_num_pixel)
                temp_num_pixel = temp_num_pixel * len(real_img) * len(real_img)
                temp_num_pixel = np.round(temp_num_pixel)
                
                # [1, 128, 128, 1]
                fake_B = A2B_gener_2D(ge_noise(temp_num_pixel), training=False)
                fake_img = (fake_B + 1) * 1.5  # [1, 128, 128, 1]
                fake_img = tf.squeeze(fake_img, -1)  # [1, 128, 128]
                fake_img = tf.squeeze(fake_img, 0)  # [128, 128]
                fake_img = np.round(fake_img)
                fake_img = np.array(fake_img)
                fake_img = img_smooth(fake_img)
                fake_imgs[:,:,img_num] = fake_img

########### 3D reconstruction
            test_3D_input = fake_imgs / 1.5 - 1
            test_3D_input = np.expand_dims(test_3D_input, 0)
            test_3D_input = np.expand_dims(test_3D_input, -1)

            # [1, 128, 128, 3]
            fake_B = A2B_gener_3D(test_3D_input, training=False)
            fake_img = (fake_B[0] + 1) * 1.5  # [128, 128, 128, 1]
            fake_img = tf.squeeze(fake_img, -1)  # [128, 128, 128]
            fake_img = np.round(fake_img)

            fake_img = np.array(fake_img)
            fmap_0 = np.where(fake_img == 0)
            fmap_1 = np.where(fake_img == 1)
            fmap_2 = np.where(fake_img == 2)
            fmap_3 = np.where(fake_img == 3)
            f_phi = len(fmap_0[0])/(len(fake_img)*len(fake_img)*len(fake_img))
            f_out = len(fmap_1[0])/(len(fake_img)*len(fake_img)*len(fake_img))
            f_in = len(fmap_2[0])/(len(fake_img)*len(fake_img)*len(fake_img))
            f_unh = len(fmap_3[0])/(len(fake_img)*len(fake_img)*len(fake_img))
            
            # calculate for pore
            fake_phi = fake_img * 0
            fake_phi[fmap_0] = 1
            f_phi_l2rec = l2_rec1_3D(fake_phi)
            f_phi_xa = np.array(f_phi_l2rec[:27,1])
            f_phi_ya = np.array(f_phi_l2rec[:27,2])
            f_phi_za = np.array(f_phi_l2rec[:27,3])
            f_phi_l2 = (f_phi_xa + f_phi_ya + f_phi_za) / 3
    
            # calculate for outer product
            fake_out = fake_img * 0
            fake_out[fmap_1] = 1
            f_out_l2rec = l2_rec1_3D(fake_out)
            f_out_xa = np.array(f_out_l2rec[:27,1])
            f_out_ya = np.array(f_out_l2rec[:27,2])
            f_out_za = np.array(f_out_l2rec[:27,3])
            f_out_l2 = (f_out_xa + f_out_ya + f_out_za) / 3

            # 비율 감별
            if abs((r_phi - f_phi) / r_phi) < pore_tor:
                pt = 0
            else:
                pt = 1
                
            if abs((r_out - f_out) / r_out) < outer_tor:
                ot = 0
            else:
                ot = 1
                
            if (sum(abs(r_out_l2 - f_out_l2) / r_out_l2) / len(r_out_l2)) < ll_tor:
                lt = 0
            else:
                lt = 1
                
            if pt + ot + lt == 0:
                print("number: {} successful".format(dir_n + 1))

                result_2D_folder = '%s/reconstructed_%03d' % (FLAGS.save_path2D, dir_n+1)
                if not os.path.isdir(result_2D_folder):
                    print("Make reconstructed_{} folder to save 2D samples".format(dir_n+1))
                    os.makedirs(result_2D_folder)
                for j in range(3):
                    fake_ = fake_imgs[:,:,j]
                    np.savetxt("{0}/fake{1:02d}_{2:02d}.txt".format(result_2D_folder, counts2+1, j+1),
                       fake_, fmt='%d', delimiter=' ')  # [128, 128]
                
                result_3D_folder = '%s/reconstructed_%03d' % (FLAGS.save_path3D, dir_n+1)
                if not os.path.isdir(result_3D_folder):
                    print("Make reconstructed_{} folder to save 3D samples".format(dir_n+1))
                    os.makedirs(result_3D_folder)
                for j in range(128):
                    fake_ = fake_img[:,:,j]  # [128, 128]
                    np.savetxt("{0}/fake{1:02d}_{2:03d}.txt".format(result_3D_folder,counts2+1 ,j+1),
                               fake_, fmt='%d', delimiter=' ')
                    
                time1 = datetime.now()
                phi_counts = 0
                out_counts = 0
                ll_counts = 0
                counts1 = 0
                counts2 += 1
            else:
                phi_counts += pt
                out_counts += ot
                ll_counts += lt

            counts1 += 1

            if counts1 % 100 == 0:
                print("phi_counts = {0} / out_counts = {1} / ll_counts = {2}".format(phi_counts, out_counts,ll_counts))
                if out_counts <= phi_counts and ll_counts <= phi_counts:
                    pore_tor += 0.01
                    print("pore_tor = {0:.01f}%".format(pore_tor*100))
                elif phi_counts <= out_counts and ll_counts <= out_counts:
                    outer_tor += 0.01
                    print("outer_tor = {0:.01f}%".format(outer_tor*100))
                else:
                    ll_tor += 0.01
                    print("ll_tor = {0:.01f}%".format(ll_tor*100))
                phi_counts = 0
                out_counts = 0
                ll_counts = 0

            time2 = datetime.now()
            if ((time2.day*60*24 + time2.hour*60 + time2.minute) - (time1.day*60*24 + time1.hour*60 + time1.minute)) > FLAGS.time_limit:         
                time_out = False
                print("number: {} breaked!!!!".format(dir_n + 1))

                
if __name__ == "__main__":
    main()