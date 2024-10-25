import os
import numpy as np
from config import FLAGS, col_val

def func_ge_A_data(data_name):
    temp_data = os.listdir(data_name)
    temp_data.sort()
    data = np.loadtxt(data_name + "/" + temp_data[0],
                      dtype=np.int32,
                      skiprows=0,
                      usecols=col_val)
    data = data.reshape(-1)
    data = np.random.permutation(data)
    data = data.reshape(FLAGS.input_shape[:2])
    data = np.expand_dims(data, -1)
        
    return data

def func_ge_B_data(data_name):
    temp_data = os.listdir(data_name)
    temp_data.sort()
    data_temp1 = np.loadtxt(data_name + "/" + temp_data[0],
                       dtype=np.int32,
                       skiprows=0,
                       usecols=col_val)
    data_temp1 = np.expand_dims(data_temp1, -1)
    
    data_temp2 = np.loadtxt(data_name + "/" + temp_data[1],
                       dtype=np.int32,
                       skiprows=0,
                       usecols=col_val)
    data_temp2 = np.expand_dims(data_temp2, -1)
    
    data = np.concatenate((data_temp1, data_temp2), axis = 2)
    
    for i in range(len(temp_data)-2):
        data_temp3 = np.loadtxt(data_name + "/" + temp_data[i + 2],
                       dtype=np.int32,
                       skiprows=0,
                       usecols=col_val)
        data_temp3 = np.expand_dims(data_temp3, -1)
        
        data = np.concatenate((data, data_temp3), axis = 2)

    return data