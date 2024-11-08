import numpy as np
import os
from config import FLAGS

def func_ge_A_data_3D(data_name):
    data_name = str(data_name)
    temp_data = os.listdir(data_name)
    temp_data.sort()

    data_temp1 = np.loadtxt(os.path.join(data_name, temp_data[0]), dtype=np.int32, skiprows=0, usecols=range(FLAGS.input_shape[0]))
    data_temp1 = np.expand_dims(data_temp1, -1)

    data_temp2 = np.loadtxt(os.path.join(data_name, temp_data[1]), dtype=np.int32, skiprows=0, usecols=range(FLAGS.input_shape[0]))
    data_temp2 = np.expand_dims(data_temp2, -1)
    
    data = np.concatenate((data_temp1, data_temp2), axis=2)
    
    for i in range(len(temp_data) - 2):
        data_temp3 = np.loadtxt(os.path.join(data_name, temp_data[i + 2]), dtype=np.int32, skiprows=0, usecols=range(FLAGS.input_shape[0]))
        data_temp3 = np.expand_dims(data_temp3, -1)
        data = np.concatenate((data, data_temp3), axis=2)
        
    return data