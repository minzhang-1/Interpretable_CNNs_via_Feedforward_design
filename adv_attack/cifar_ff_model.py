import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
import pickle
import numpy as np
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense,Lambda,MaxPooling2D

data_dir='cifar_ff_model/'

with open(data_dir+'pca_params.pkl','rb') as fr:
    pca_params=pickle.load(fr, encoding='latin1')
fr=open(data_dir+'llsr_weights.pkl','rb')
weights=pickle.load(fr)
fr.close()
fr=open(data_dir+'llsr_bias.pkl','rb')
biases=pickle.load(fr)
fr.close()

with open(data_dir+'std_var.pkl','rb') as fr:
    std_var=pickle.load(fr)

def weight_init_conv_1(shape, dtype=None):
#     print('shape: ', shape)
    weight=pca_params['Layer_0/kernel'].astype(np.float32)
    weight=np.moveaxis(weight, 1,0)
    return weight.reshape(shape)


def weight_init_conv_2(shape, dtype=None):
    weight=pca_params['Layer_1/kernel'].astype(np.float32)
    weight=np.moveaxis(weight, 1,0)
    return weight.reshape(shape)


def bias_init_conv_2(shape, dtype=None):
    weight=np.transpose(pca_params['Layer_1/kernel'].astype(np.float32))
    bias=np.zeros(800)
    bias=bias+1/np.sqrt(800)*pca_params['Layer_1/bias'].astype(np.float32)

    bias1=np.matmul(bias, weight)

    bias2 = np.zeros(shape, dtype=np.float32)
    bias2[0] = -1
    bias2 = bias2 * pca_params['Layer_1/bias'].astype(np.float32)

    bias_final= bias1+ bias2
    return bias_final.reshape(shape)


def weight_init_fc_1(shape, dtype=None):
    return weights['0 LLSR weight'].reshape(shape)


def bias_init_fc_1(shape, dtype=None):
    return biases['0 LLSR bias'].reshape(shape)


def weight_init_fc_2(shape, dtype=None):
    return weights['1 LLSR weight'].reshape(shape)


def bias_init_fc_2(shape, dtype=None):
    return biases['1 LLSR bias'].reshape(shape)


def weight_init_fc_3(shape, dtype=None):
    return weights['2 LLSR weight'].reshape(shape)


def bias_init_fc_3(shape, dtype=None):
    return biases['2 LLSR bias'].reshape(shape)

def cifar_ff_model():
    model = Sequential()
    model.add(
        Conv2D(32, kernel_size=(5, 5), activation=None, data_format='channels_last', input_shape=(32,32,3),
               kernel_initializer=weight_init_conv_1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (5, 5), activation=None, data_format='channels_last', kernel_initializer=weight_init_conv_2,
                     bias_initializer=bias_init_conv_2))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Lambda(lambda x: x / std_var))
    model.add(Activation('relu'))
    model.add(Dense(200, kernel_initializer=weight_init_fc_1, bias_initializer=bias_init_fc_1))
    model.add(Activation('relu'))
    model.add(Dense(100, kernel_initializer=weight_init_fc_2, bias_initializer=bias_init_fc_2))
    model.add(Activation('relu'))
    model.add(Dense(10, kernel_initializer=weight_init_fc_3, bias_initializer=bias_init_fc_3))
    model.add(Activation('softmax'))
    return model