# University of Victoria
# CSC 421
# Assignment 1
# K - Nearest Neighbors Algorithm

# Name: Tarush Roy
# StudentID: V00883469

import numpy as np
from keras.datasets import cifar10

import time

# debug switch
debug = False

# Takes data and returns labeled data
def predict(test_data, training_data, training_labels, k):
    predicted_labels = []

    for i in range(len(test_data)):
        if debug:
            t1 = time.time()
        
        # get distances between test data point and all training data
        distances = [np.linalg.norm(test_data[i] - train_data_image) for train_data_image in training_data]

        if debug:
            t2 = time.time()
            print("time taken = " + str(t2-t1))

        # sort distances
        sorted_indexes = np.argsort(distances)

        # get k nearest neighbors by getting first k labels
        nn_labels = []
        for j in range(k):
            nn_labels.append(training_labels[sorted_indexes[j]])

        # get most common label number
        predicted_label = np.argmax(np.bincount(nn_labels))
        predicted_labels.append(predicted_label)
    
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
    # load data
    (train_data, train_labels), (test_data, test_labels) = cifar10.load_data()
    
    # reshape and flatten data
    train_data = train_data.reshape(50000, -1).astype('float')
    train_labels = train_labels.reshape(50000, -1).astype('float').flatten()

    test_data = test_data.reshape(10000, -1).astype('float')
    test_labels = test_labels.reshape(10000, -1).astype('float').flatten()

    # hyperparameters k -> number of neighbors
    k = 11

    # predict
    predicted_labels = predict(test_data, train_data, train_labels, k)

    # evaluate
    accuracy = evaluate(predicted_labels, test_labels)
    print("final accuracy on main test set = " + str(accuracy * 100) + "% with k = " + str(k))
