import math
import csv
import numpy as np

ITERATIONS = 100
LEARNING_RATE = 0.00625

def read_file(file):
    # helper function to read through files
    data = np.genfromtxt(file, delimiter = ',', dtype=int, skip_header=1)
    features = data[: , :-2]
    labels = data[:, -1:]
    features = np.insert(features, 0, 1, axis=1)
    return features, labels


def sigmoid(z):
    return 1/(1+math.exp(-z))

def train(features,labels, learning_rate):
    # training function which applies logistic regression model
    m = len(features[0])
    theta = np.zeros(m)
    # loops for number of iterations specified and updates theta & gradient
    for i in range(ITERATIONS):
        gradient = np.zeros(m)
        for (x, y) in zip(features, labels):
            for j in range(len(x)):
                gradient[j] += x[j] * (y[0] - (sigmoid(np.dot(theta, x))))
        theta += learning_rate * gradient
    return theta

def test(file, theta):
    # sifts through test file data and applies ML algorithm to it, and tracks actual results for future comparisons
    test_features, test_results = read_file(file)
    predictions = []
    actual = []
    # loops through features & applies classification
    for elem in test_features:
        probability = sigmoid(np.dot(theta,elem))
        if probability > 0.5:
            predictions.append(1)
        else:
            predictions.append(0)
    # tracks true results
    for elem in test_results:
        actual.append(elem[0])
    # returns ML classifications, and true results
    return predictions,actual

def log_liklihood(theta, features, labels):
    # helper function to calculate log-liklihood of data if desired
    total_sum = 0
    for (x, y) in zip(features, labels):
        for j in range(len(x)):
            total_sum += (y[0] * (math.log(sigmoid(0)))) +(((1-y[0]) *math.log((1- sigmoid(0)))))
    return total_sum


def main():
    # small data training file -- could be expanded
    train_file = open("/Users/yousifmohsen/PycharmProjects/machine_learning//netflix-small-train.csv")
    features, labels = read_file(train_file)
    # specify learning rate constant
    theta = train(features,labels,LEARNING_RATE)
    test_file = open("/Users/yousifmohsen/PycharmProjects/machine_learning//netflix-test.csv")
    result, actual = test(test_file,theta)
    count = 0
    # compare classifications to actual
    for i in range(len(result)):
        if result[i] == actual[i]:
            count += 1
    # find log-liklihood
    total_sum = log_liklihood(theta,features,labels)
    # print classification accuracy
    print(count/len(result))


if __name__ == "__main__":
    main()