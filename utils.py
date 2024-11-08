import numpy as np
from skimage import measure
from config import FLAGS

def ge_noise(num_pixels):
    count_0 = int(num_pixels[0])
    count_2 = int(num_pixels[2])
    count_3 = int(num_pixels[3])
    A_data = np.ones(FLAGS.input_shape[0] * FLAGS.input_shape[0])
    A_data[0:count_0] = 0
    A_data[count_0:count_0 + count_2] = 2
    A_data[count_0 + count_2:count_0 + count_2 + count_3] = 3
    A_data = A_data / 1.5 - 1
    for i in range(2):
        A_data = np.random.permutation(A_data)
    A_data = A_data.reshape(FLAGS.input_shape[:2])
    A_data = [A_data]
    A_data = np.expand_dims(A_data, -1)
    
    return A_data

def img_smooth(fake_img):
    index_3 = fake_img - 3
    map_3 = np.where(index_3 != 0)
    index_3[map_3] = 1
    all_labels_3 = measure.label(index_3)
    map_3_2 = np.where(all_labels_3 > 1)
    fake_img[map_3_2] = 3

    index_0 = fake_img * -1
    map_0 = np.where(index_0 != 0)
    index_0[map_0] = 1
    all_labels_0 = measure.label(index_0)
    map_0_2 = np.where(all_labels_0 > 1)
    fake_img[map_0_2] = 0
    
    return fake_img

def l2_rec1_3D(data):

    # phase information
    # phase 0: void
    # phase 1: solid
    data = np.array(data)
    l = len(data)
    l2rec1 = np.zeros((l,4))  # p2 data storage

    # for x-direction (matrix +1)----------------------------
    xl = np.zeros((l-1,2)) # cluster fn. for x-dir
    xl[:,0] = np.array(range(1,l))

    data = -1 * (data - 1)    # chage phase 0=solid 1=void
    data2 = np.array(data + 1 - 1)

    for t in range(1,l):
        data2[t,:,:] = data2[t,:,:] * (data2[t-1,:,:] + data2[t,:,:])

    for r in range(1,l):
        data3 = data2 - r
        data3 = np.array(data3 + abs(data3)) / 2
        pick_size = (l - r) * l * l
        xl[r-1,1] = ((l * l * l) - len(np.where(data3 == 0)[0])) / float(pick_size)
        if xl[r-1,1] == 0:
            break

    vf = float(sum(sum(sum(data))))
    vf /= float(l * l * l)

    l2rec1[0,1:4] = vf
    l2rec1[1:l,0] = xl[:,0] / (l - 1)
    l2rec1[1:l,1] = xl[:,1]

    # for y-direction (matrix +1)----------------------------
    yl = np.zeros((l-1,2)) # cluster fn. for y-dir
    yl[:,0] = np.array(range(1,l))

    data2 = np.array(data + 1 - 1)

    for t in range(1,l):
        data2[:,t,:] = data2[:,t,:] * (data2[:,t-1,:] + data2[:,t,:])

    for r in range(1,l):
        data3 = data2 - r
        data3 = np.array(data3 + abs(data3)) / 2
        pick_size = (l - r) * l * l
        yl[r-1,1] = ((l * l * l) - len(np.where(data3 == 0)[0])) / float(pick_size)
        if yl[r-1,1] == 0:
            break

    l2rec1[1:l,2] = yl[:,1]
    
    # for z-direction (matrix +1)----------------------------
    zl = np.zeros((l-1,2)) # cluster fn. for z-dir
    zl[:,0] = np.array(range(1,l))

    data2 = np.array(data + 1 - 1)

    for t in range(1,l):
        data2[:,:,t] = data2[:,:,t] * (data2[:,:,t-1] + data2[:,:,t])

    for r in range(1,l):
        data3 = data2 - r
        data3 = np.array(data3 + abs(data3)) / 2
        pick_size = (l - r) * l * l
        zl[r-1,1] = ((l * l * l) - len(np.where(data3 == 0)[0])) / float(pick_size)
        if zl[r-1,1] == 0:
            break

    l2rec1[1:l,3] = zl[:,1]
    
    return l2rec1