# University of Victoria
# CSC 421
# Assignment 1
# K - Means Algorithm

# Name: Tarush Roy
# StudentID: V00883469

import numpy as np
from keras.datasets import cifar10

import time

# assigns data points to a cluster randomly
def cluster(training_set, k):
    clusters = [[] for _ in range(k)]

    # assign one random data point to each cluster
    points = []
    np.random.shuffle(training_set)
    for i in range(k):
        clusters[i].append(training_set[i])
        points.append(training_set[i])

    # assign each data point to closest cluster
    for data_point in training_set:
        for point in points:
            if(np.array_equal(data_point, point)):
                break
        # measure distance to each cluster from data point
        distances = [np.linalg.norm(data_point[:3072] - point[:3072]) for point in points]
        # get index of nearest cluster
        cluster_index = np.argmin(distances)
        # add data point to nearest cluster
        clusters[cluster_index].append(data_point)

    return clusters

# takes np array, example: (10, 4000, 3073) 
# 10 clusters, 4000 images each cluster, 3072 pixel values each image, 1 label each image
def train(clusters, training_set, k, old_means):
    # new clusters
    new_clusters = [[] for _ in range(k)]

    # compute mean of each cluster
    means = np.array([np.mean(cluster, axis=0)[:3072] for cluster in clusters])

    # reassign each data point to closest cluster
    for data_point in training_set:
        # measure distance to each cluster from data point
        distances = [np.linalg.norm(data_point[:3072] - mean[:3072]) for mean in means]
        # get index of nearest cluster
        cluster_index = np.argmin(distances)
        # add data point to nearest cluster
        new_clusters[cluster_index].append(data_point)

    # repeat until means dont change
    if(np.array_equal(means, old_means)):
        return (new_clusters, means)
    else:
        return train(new_clusters, training_set, k, means)

# Takes cluster and test points and returns labeled data
def predict(clusters, means, test_set):
    predicted_labels = []
    for test_data in test_set:
        # measure distance to each cluster from data point
        distances = [np.linalg.norm(test_data[:3072] - mean[:3072]) for mean in means]
        # get index of nearest cluster
        cluster_index = np.argmin(distances)
        # get most common label from that cluster and add to predicted labels
        labels = np.split(np.array(clusters[cluster_index]), len(clusters[cluster_index][0]), axis=1)[-1].flatten().tolist()
        predicted_label = np.argmax(np.bincount(labels))
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

def k_fold(training_set, k):
    # k-fold k
    kfcv = 5

    # spliit data into k folds
    split_set = np.split(training_set, kfcv)

    # list of accuracies for each validation set
    accuracies = []

    # pick validation set and split into training and validation sets
    # predict and evaluate
    for i in range(kfcv):
        # ith set is validation set
        validation_set = split_set[i]
        validation_labels = np.split(validation_set, len(validation_set[0]), axis=1)[-1].flatten()

        # combine all other data sets 
        train_set = []
        for j in range(kfcv):
            if(j != i):
                for l in range(len(split_set[j])):
                    train_set.append(split_set[j][l])
        
        # convert to numpy array
        train_set = np.array(train_set)

        # cluster training set
        random_clusters = np.array(cluster(train_set, k), dtype=object)

        print('training ...')

        # train (re-cluster)
        clusters, means = train(random_clusters, train_set, k, [])

        print('training complete.')
        print('predicting ...')

        # predict using cluster model and validation data
        predicted_labels = predict(clusters, means, validation_set)

        # evaluate
        accuracy = evaluate(predicted_labels, validation_labels)
        print('accuracy: ' + str(accuracy))
        accuracies.append(accuracy)
    
    # calc avg and return it
    return np.mean(np.array(accuracies))

if __name__ == "__main__":
    # load data
    (train_data, train_labels), (test_data, test_labels) = cifar10.load_data()
    
    # reshape and flatten data
    train_data = train_data.reshape(50000, -1).astype('float')
    train_labels = train_labels.reshape(50000, -1).astype('float')

    train_set = np.concatenate((train_data, train_labels), axis=1)

    test_data = test_data.reshape(10000, -1).astype('float')
    test_labels = test_labels.reshape(10000, -1).astype('float')

    test_set = np.concatenate((test_data, test_labels), axis=1)

    # hyperparameters k -> number of clusters
    #k = [3, 5, 7, 11, 10]
    k = [10]

    # run k-fold cross validation
    avg_acc = [k_fold(train_set, k_val) for k_val in k]

    # find best k 
    max_k_i = np.argmax(avg_acc)
    print("max accuracy during k-fold: " + str(avg_acc[max_k_i]))

    # use best k to run on main test and train data
    # cluster training set
    random_clusters = np.array(cluster(train_set, k[max_k_i]), dtype=object)

    print('training ...')

    # train (re-cluster)
    clusters, means = train(random_clusters, train_set, k[max_k_i], [])

    print('training complete.')
    print('predicting ...')

    # predict using cluster model and validation data
    predicted_labels = predict(clusters, means, test_set)

    # evaluate
    test_labels = test_labels.flatten()
    accuracy = evaluate(predicted_labels, test_labels)
    print('final accuracy: ' + str(accuracy))
