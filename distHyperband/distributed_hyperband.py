import glob
import keras
import os
import numpy as np
import random
import shortuuid
import time
from keras import optimizers
from keras.datasets import mnist
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import MaxPooling2D
from keras.models import load_model
from keras.models import Sequential
from mpi4py import MPI
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

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
    filename='./modelSaved/'+shortuuid.uuid()+'.h5'
    model.save(filename)

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
    num_tasks = len(hyperparameters_list)
    random.shuffle(hyperparameters_list)

    status = MPI.Status()

    # Initial hyperparameter distribution
    for i in range(1, size):
        if len(hyperparameters_list) == 0:
            break
        data = hyperparameters_list[-1]
        hyperparameters_list = hyperparameters_list[:-1]
        comm.send(data, dest=i, tag=0)

    results = []
    completed = 0

    # Receive results and distribute remaining hyperparameters
    while len(hyperparameters_list) > 0 or completed < num_tasks:
        performance = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        results.append(performance)
        completed += 1
        print(f'MAIN - Received from node {status.Get_source()}')
        print(performance)

        if len(hyperparameters_list) > 0:
            comm.send(hyperparameters_list[-1], dest=status.Get_source(), tag=0)
            print(f'MAIN - Sent to node {status.Get_source()}')
            hyperparameters_list = hyperparameters_list[:-1]

    # Send termination signal.
    for i in range(1, size):
        comm.send(None, dest=i, tag=0)

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    path = "./modelSaved/"
    #for file in glob.glob(os.path.join(path, '*')):
    #    os.remove(file)

    h5_files = glob.glob(os.path.join(path, '*.h5'))

    # Creating deep ensemble
    # TODO - distribute model loading and predicting with MPI
    mc_predictions = [load_model(file).predict(x_test, batch_size=1000) for file in h5_files]

    p = np.array(mc_predictions)
    y_mean = p.mean(axis=0)
    w = 1 / np.sum(y_mean, axis=1, keepdims=True)
    y_mean *= w
    #y_std = p.std(axis=0)*w

    # Calc brier score
    y_test_one_hot = np.eye(10)[y_test]
    brier_scores = [brier_score_loss(y_test_one_hot[:, i], y_mean[:, i]) for i in range(10)]
    print(f"Brier scores for each class: {brier_scores}")
    mean_bs = np.mean(np.mean((y_mean - y_test_one_hot) ** 2, axis = 1))
    print(f"Mean Brier score: {mean_bs}")

    # Calculate calibration error
    n_bins = 10
    bin_limits = np.linspace(0, 1, n_bins + 1)
    bin_mids = (bin_limits[1:] + bin_limits[:-1]) / 2
    y_prob_max = y_mean[np.arange(y_mean.shape[0]), y_mean.argmax(axis=1)]
    bin_indices = np.digitize(y_prob_max, bin_limits) - 1
    bin_counts = np.bincount(bin_indices, minlength=n_bins)
    bin_sums = np.bincount(bin_indices, weights=(y_mean.argmax(axis=1) == y_test), minlength=n_bins)
    non_empty_bins = bin_counts > 0
    avg_pred_prob = np.bincount(bin_indices, weights=y_prob_max, minlength=n_bins) / bin_counts
    bin_acc = bin_sums / bin_counts
    calibration_error = np.sum(bin_counts[non_empty_bins] * np.abs(avg_pred_prob[non_empty_bins] - bin_acc[non_empty_bins])) / np.sum(bin_counts)
    print(f"Calibration Error: {calibration_error}")

else:
    # Worker nodes
    while True:
        hyperparameters = comm.recv(source=0, tag=MPI.ANY_TAG)
        if hyperparameters is None:
            #print(f'Node {rank} - DONE')
            break
        #print(f'Node {rank} - Starting {hyperparameters}')
        performance = digitsWithHPO("hyperband", *hyperparameters)
        #print(f'Node {rank} - Completed Performance: ', performance)
        comm.send(performance, dest=0, tag=0)

