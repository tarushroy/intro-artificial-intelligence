# University of Victoria
# CSC 421
# Assignment 1
# K - Nearest Neighbors Algorithm

# Name: Tarush Roy
# StudentID: V00883469

import numpy as np
from keras.datasets import cifar10

import time

# l1 distance metric
# takes 2 image vectors (numpy array) as input
def l1distance(x, y):
    return np.sum(x - y)

# euclidean distance between any 2 images
# takes 2 image vectors (numpy array) as input
def l2distance(x, y):
    return np.sqrt(np.sum(np.square(x - y)))

# Takes data and returns labeled data
def predict(test_data, training_data, training_labels, k):
    predicted_labels = []

    for i in range(len(test_data)):
        tic = time.time()

        # get distances between test data point and all training data
        distances = [np.linalg.norm(test_data[i] - train_data_image) for train_data_image in training_data]

        toc = time.time()
        print("time taken: " + str(toc-tic))

        # sort distances
        sorted_indexes = np.argsort(distances)

        # get k nearest neighbors by getting first k labels
        nn_labels = [training_labels[index] for index in sorted_indexes]

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

# takes training data and training labels
# runs k-fold with all possible validation sets
# returns average accuracy
def k_fold(training_data, training_labels, k):
    # k-fold k
    kfcv = 5

    # spliit data into k folds
    split_data = np.split(training_data, kfcv)
    split_labels = np.split(training_labels, kfcv)

    # list of accuracies for each validation set
    accuracies = []

    # pick validation set and split into training and validation sets
    # predict and evaluate
    for i in range(kfcv):
        # ith set is validation set
        validation_data = split_data[i]
        validation_labels = split_labels[i]

        # combine all other data sets
        train_data = []
        train_labels = []
        for j in range(kfcv):
            if(j != i):
                for l in range(len(split_data[j])):
                    train_data.append(split_data[j][l])
                    train_labels.append(split_labels[j][l])
        
        # predict
        predicted_labels = predict(validation_data, train_data, train_labels, k)

        # evaluate
        accuracies.append(evaluate(predicted_labels, validation_labels))
        print("accuracy: " + str(accuracies[i]))
    
    # calc avg and return it
    return np.mean(np.array(accuracies))
        

if __name__ == "__main__":
    # load data
    (train_data, train_labels), (test_data, test_labels) = cifar10.load_data()
    
    # reshape and flatten data
    train_data = train_data.reshape(50000, -1).astype('float')
    train_labels = train_labels.reshape(50000, -1).astype('float').flatten()

    test_data = test_data.reshape(10000, -1).astype('float')
    test_labels = test_labels.reshape(10000, -1).astype('float').flatten()

    # hyperparameters k -> number of neighbors
    k = [3, 5, 7, 11]

    # run k-fold cross validation
    avg_acc = [k_fold(train_data, train_labels, k_val) for k_val in k]

    # find best k
    max_k_i = np.argmax(avg_acc)
    print("max accuracy during k-fold: " + str(avg_acc[max_k_i]))

    # use best k to run on main test and train data
    # predict
    predicted_labels = predict(test_data, train_data, train_labels, k[max_k_i])

    # evaluate
    accuracy = evaluate(predicted_labels, test_labels)

    print("final accuracy of knn with k-fold cross validation: " + str(accuracy))