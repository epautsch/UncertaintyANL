import keras
import random
import time
from keras import optimizers
from keras.datasets import mnist
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import MaxPooling2D
from keras.models import Sequential
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def digitsWithHPO(tuner, lr, mo, ep):
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
    model.compile(loss=keras.losses.categorical_crossentropy,
        optimizer=optimizers.Adam(learning_rate=lr),
        metrics=['accuracy'])

    #train
    hist = model.fit(x_train, y_train,batch_size=batch_size,epochs=ep,verbose=0,validation_data=(x_test, y_test))
    #filename='./modelSaved/'+shortuuid.uuid()+'.h5'
    #model.save(filename)

    #evaluate
    score = model.evaluate(x_test, y_test, verbose=0)
    #print('Test loss: ', score[0])
    #print('Test Acceracy: ', score[1])
    return score

# Main Node
if rank == 0:
    hyperparameters_list = [[0.1, 0.2, 1],
         [0.1, 0.2, 20],
         [0.1, 0.2, 100],
         [0.1, 0.2, 200],
         [0.01, 0.2, 1],
         [0.01, 0.2, 20],
         [0.01, 0.2, 100],
         [0.01, 0.2, 200],
         [0.001, 0.2, 1],
         [0.001, 0.2, 20],
         [0.001, 0.2, 100],
         [0.001, 0.2, 200],
         [0.0001, 0.2, 1],
         [0.0001, 0.2, 20],
         [0.0001, 0.2, 100],
         [0.0001, 0.2, 200]]
    random.shuffle(hyperparameters_list)

    status = MPI.Status()

    # Initial hyperparameter distribution
    for i in range(1, size):
        if len(hyperparameters_list) == 0:
            break
        data = hyperparameters_list[-1]
        hyperparameters_list = hyperparameters_list[:-1]
        comm.send(data, dest=i, tag=0)

    # Still have hyperparameters left.
    while len(hyperparameters_list) > 0:
        performance = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        print(f'MAIN - Received from node {status.Get_source()}')
        comm.send(hyperparameters_list[-1], dest=status.Get_source(), tag=0)
        print(f'MAIN - Sent to node {status.Get_source()}')
        hyperparameters_list = hyperparameters_list[:-1]

    # If there's no hyperparameters left, send a termination signal.
    for i in range(1, size):
        comm.send(None, dest=i, tag=0)

else:
    # Worker nodes
    while True:
        hyperparameters = comm.recv(source=0, tag=MPI.ANY_TAG)
        if hyperparameters is None:
            print(f'Node {rank} - DONE')
            break
        print(f'Node {rank} - Starting {hyperparameters}')
        performance = digitsWithHPO("hyperband", *hyperparameters)
        print(f'Node {rank} - Completed Performance: ', performance)
        comm.send(performance, dest=0, tag=0)

