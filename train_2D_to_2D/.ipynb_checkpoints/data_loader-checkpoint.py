import numpy as np
import os
from config import FLAGS, col_val

def func_ge_B_data(data_name):

    data = np.loadtxt(data_name,
                       dtype=np.int32,
                       skiprows=0,
                       usecols=col_val)

    data = np.array(data, np.float32)

    return data


def func_ge_A_data(data):

    data2 = data.reshape(-1)
    for i in range(2):
        data2 = np.random.permutation(data2)
    data2 = data2.reshape(FLAGS.input_shape[:2])

    return data2