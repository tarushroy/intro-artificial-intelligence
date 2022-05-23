# University of Victoria
# CSC 421
# Assignment 1
# K - Nearest Neighbors Algorithm

# Name: Tarush Roy
# StudentID: V00883469

import os
import time
import numpy as np
import pickle

# Taken from https://www.cs.toronto.edu/~kriz/cifar.html
# Reference:
# Learning Multiple Layers of Features from Tiny Images, Alex Krizhevsky, 2009.
# https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# l1 distance metric
# takes 2 image vectors (numpy array) as input
def l1distance(x, y):
    return np.sum(x - y)

# euclidean distance between any 2 images
# takes 2 image vectors (numpy array) as input
def l2distance(x, y):
    return np.sqrt(np.sum((x - y) ** 2))

# Takes data and returns labeled data
def predict(test_data, training_data, training_labels, k):
    predicted_labels = []

    for i in range(len(test_data)):
        #t0 = time.time()

        distances = []

        # get distance between test data point and all training data
        for j in range(len(training_data)):
            # distances.append(l2distance(test_data[i], training_data[j]))
            distances.append(np.linalg.norm(test_data[i] - training_data[j]))

        # sort distances
        sorted_indexes = np.argsort(distances)

        # get k nearest neighbors by getting first k labels
        nn_labels = []
        for j in range(k):
            nn_labels.append(training_labels[sorted_indexes[j]])

        # get most common label number
        predicted_label = np.argmax(np.bincount(nn_labels))
        predicted_labels.append(predicted_label)

        #t1 = time.time()
        #print("time elapsed for 1 test vector: " + str(t1-t0))
    
    return predicted_labels


# Takes labeled data and ground truth labels
def evaluate(predicted_labels, true_labels):   
    size_of_dataset = len(predicted_labels)
    number_of_correct_predictions = 0

    for i in range(size_of_dataset):
        if(predicted_labels[i] == true_labels[i]):
            number_of_correct_predictions += 1

    return number_of_correct_predictions/size_of_dataset

if __name__ == "__main__":
    # Copied, must be changed
    fileDir = os.path.dirname(os.path.realpath('__file__'))

    # hyperparameter k
    k = [3, 5, 7, 11]

    # load label names from batches.meta
    label_names = unpickle(os.path.join(fileDir, '../data/batches.meta'))
    label_names = [label.decode() for label in label_names[b'label_names']]

    # Load data
    data_batch_1 = unpickle(os.path.join(fileDir, '../data/data_batch_1'))
    data_batch_2 = unpickle(os.path.join(fileDir, '../data/data_batch_2'))
    data_batch_3 = unpickle(os.path.join(fileDir, '../data/data_batch_3'))
    data_batch_4 = unpickle(os.path.join(fileDir, '../data/data_batch_4'))
    data_batch_5 = unpickle(os.path.join(fileDir, '../data/data_batch_5'))

    data = [data_batch_1, data_batch_2, data_batch_3, data_batch_4, data_batch_5]

    # run k-fold cross validation to find best k (hyperparameter)
    kfcv = 5

    '''
    for i in range(kfcv):
        # ith data batch is the validation set
        validation_data = data[i][b'data']
        validation_labels = data[i][b'labels']
        # combine all other data sets
        training_data = []
        training_labels = []
        for j in range(kfcv):
            if(j != i):
                for k in range(len(data[j][b'data'])):
                    training_data.append(data[j][b'data'][k])
                    training_labels.append(data[j][b'labels'][k])
        
        # predict
        #tic = time.time()
        predicted_labels = predict(validation_data, training_data, training_labels, k)
        #toc = time.time()
        #print("time taken for 1 k-fold prediction label set = " + str(toc-tic))
        # evaluate
        accuracy = evaluate(predicted_labels, validation_labels)
        print("accuracy with " + str(i) + "th data set as validation set = " + str(accuracy) + '\n')
    '''

    # load test data
    test_batch = unpickle(os.path.join(fileDir, '../data/test_batch'))
    test_data = test_batch[b'data']
    test_labels = test_batch[b'labels']

    # combine data
    training_data = []
    training_labels = []
    for i in range(5):
        for j in range(len(data[i][b'data'])):
            training_data.append(data[i][b'data'][j])
            training_labels.append(data[i][b'labels'][j])

    print(np.shape(training_data))

    # predict
    predicted_labels = predict(test_data, training_data, training_labels, k[0])

    # evaluate
    accuracy = evaluate(predicted_labels, test_labels)
    print("accuracy = " + str(accuracy))