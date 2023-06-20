from mpi4py import MPI
import numpy as np
import sys
import json
import keras
import tensorflow as tf
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from tensorflow.keras import optimizers
from keras import backend as K
import argparse
import shortuuid


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

hyperparam_configs = [
        {'t': 'hyperband', 'ep': 10, 'lr': 0.1, 'mo': 0.2}, # 1
        {'t': 'hyperband', 'ep': 20, 'lr': 0.1, 'mo': 0.2}, # 2
        {'t': 'hyperband', 'ep': 100, 'lr': 0.1, 'mo': 0.2}, # 3
        {'t': 'hyperband', 'ep': 200, 'lr': 0.1, 'mo': 0.2}, # 4
        {'t': 'hyperband', 'ep': 10, 'lr': 0.01, 'mo': 0.2}, # 5
        {'t': 'hyperband', 'ep': 20, 'lr': 0.01, 'mo': 0.2}, # 6
        {'t': 'hyperband', 'ep': 100, 'lr': 0.01, 'mo': 0.2}, # 7
        {'t': 'hyperband', 'ep': 200, 'lr': 0.01, 'mo': 0.2}, # 8
        {'t': 'hyperband', 'ep': 10, 'lr': 0.001, 'mo': 0.2}, # 9
        {'t': 'hyperband', 'ep': 20, 'lr': 0.001, 'mo': 0.2}, # 10
        {'t': 'hyperband', 'ep': 100, 'lr': 0.001, 'mo': 0.2}, # 11
        {'t': 'hyperband', 'ep': 200, 'lr': 0.001, 'mo': 0.2}, # 12
        {'t': 'hyperband', 'ep': 10, 'lr': 0.0001, 'mo': 0.2}, # 13
        {'t': 'hyperband', 'ep': 20, 'lr': 0.0001, 'mo': 0.2}, # 14
        {'t': 'hyperband', 'ep': 100, 'lr': 0.0001, 'mo': 0.2}, # 15
        {'t': 'hyperband', 'ep': 200, 'lr': 0.0001, 'mo': 0.2}, # 16
]

n_configs = len(hyperparam_configs)
n_configs_per_rank = n_configs // size
print(f'n_configs: {n_configs}\nsize: {size}\nn_configs_per_rank (n_configs // size): {n_configs_per_rank}')

start = rank * n_configs_per_rank
end = (rank + 1) * n_configs_per_rank if rank != size - 1 else n_configs
print(f'start: {start}\nend: {end}')

for i in range(start, end):
    config = hyperparam_configs[i]
    args = argparse.Namespace(**config)

    # construct the argument parser and parse the arguments
    #ap = argparse.ArgumentParser()
    #ap.add_argument("-t", "--tuner", required=True, type=str,
     #   choices=["hyperband", "random", "bayesian"],
      #  help="type of hyperparameter tuner we'll be using")
    #ap.add_argument("-p", "--plot", required=True,
    #	help="path to output accuracy/loss plot")
    #ap.add_argument("-lr", "--learningrate", required=False,
     # help="value for learningrate")
    #ap.add_argument("-mo", "--momentum", required=False,
     # help="value for momentum")
    #ap.add_argument("-ep", "--epochs", required=False,
     # help="value for epochs")
    #args = vars(ap.parse_args())



    #default lr is 0.01
    if float(args.lr) <= 0:
      learningrate=0.01
    else:
      learningrate=float(args.lr)
    #default momentum is 0.1
    if float(args.mo) <= 0:
      momentum=0.1
    else:
      momentum=float(args.mo)
    #defaul epochs is 2
    if float(args.ep) <= 0:
      epochs=2
    else:
      epochs=int(args.ep)

    #  to split the data of training and testing sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    #Data Preprocessing
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    input_shape = (28, 28, 1)
    # conversion of class vectors to matrices of  binary class 
    batch_size = 128
    num_classes = 10


    y_train = keras.utils.np_utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.np_utils.to_categorical(y_test, num_classes)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    #create model
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    #model.add(Dense(256, activation='relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    #model.compile(loss=keras.losses.categorical_crossentropy,
    #            optimizer=tf.keras.optimizers.SGD(learning_rate=learningrate, momentum=momentum),
    #            metrics=['accuracy'])
    model.compile(loss=keras.losses.categorical_crossentropy,
          optimizer='adam',
          metrics=['accuracy'])
    #train
    hist = model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,verbose=0,validation_data=(x_test, y_test))
    #print("The model has successfully trained")
    filename='./modelSaved/'+shortuuid.uuid()+'.h5'
    model.save(filename)
    #print("Saving the bot as mnist.h5")

    #evaluate
    score = model.evaluate(x_test, y_test, verbose=0)
    #print('Test loss: ', score[0])
    print('Test Accuracy: ', score[1])

    #print model parameters and Test Accuracy to a txt file
    #since several files will write needs lock here
    #TODO Add model parameters to file so maximun can be found
    #and who produced the max
    #add error checking here
    with open(f"output_rank_{rank}.txt", "a") as outfile:
        outfile.write(str(score[1]))
        outfile.write('\n')
