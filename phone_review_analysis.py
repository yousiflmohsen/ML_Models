import math
import csv
import numpy as np
import pandas

ITERATIONS = 100

# function to parse through the phone_reviews file in order to assess negative word usage in comparison to ratings
def read_file(file, positive_file, negative_file):
    data = pandas.read_csv(file)
    data = data.to_numpy()
    # body of review
    body = (data[:, 6])
    count = 0
    # rating given
    rating = (data[:, 2])
    # parses most used positive words into a list
    positive_words = pandas.read_csv(positive_file)
    positive_words = positive_words.to_numpy()
    positive_words = positive_words[:, 0]
    # parses most used negative words into a list
    negative_words = pandas.read_csv(negative_file)
    negative_words = negative_words.to_numpy()
    negative_words = negative_words[:, 0]
    # classified rating
    assumed_labels = []
    # actual rating
    actual_labels = []
    # loop through the ratings and reviews
    for (actual, title) in zip(rating, body):
        # assess whether a positive or negative connotation is applied, and classified as such
        positive_connotation = set(positive_words).intersection((str(title).lower()).split(" "))
        negative_connotation = set(negative_words).intersection((str(title).lower()).split(" "))

        if negative_connotation:
            assumed_labels.append(1)
        else:
            assumed_labels.append(0)
        # checks actual rating and appends to actual rating list
        if int(actual) >= 3:
            actual_labels.append(0)
        else:
            actual_labels.append(1)

    return assumed_labels, actual_labels, positive_words, negative_words

def read_test_file(file, positive_words, negative_words):
    # similar function to read file, but adjusts frames for parsing based on this file, and filters to only english reviews
    data = pandas.read_csv(file, engine='python')
    data = data.to_numpy()
    # body of review
    rating = (data[:, 6])
    rating = rating.astype(float)

    count = 0
    titles = (data[:, 8])
    total = (data[:, 7].astype(float))
    language = (data[:,2])
    assumed_labels = []
    actual_labels = []
    # neg_count = 0
    # neutral = 0
    for (actual, title, max_score, lang) in zip(rating, titles, total, language):
        #checks to ensure the language is in english
        if lang == "en":
            # classify based on connotation
            positive_connotation = set(positive_words).intersection((str(title).lower()).split(" "))
            negative_connotation = set(negative_words).intersection((str(title).lower()).split(" "))

            if negative_connotation:
                assumed_labels.append(1)
            else:
                assumed_labels.append(0)
            # append actual classifications
            if (float(actual)/float(max_score)) >= .5:
                actual_labels.append(0)
            else:
                actual_labels.append(1)

    return assumed_labels, actual_labels

def sigmoid(z):
    return 1/(1+math.exp(-z))

def train(features,labels, learning_rate):
    #m is len features[0], in this case one feature is investigated so 1
    m = 1
    theta = np.zeros(m)
    # loops based on specified iterations
    for i in range(ITERATIONS):
        #updated gradient
        gradient = np.zeros(m)
        for (x, y) in zip(features, labels):
            # range depends on number of features assessed -- in this case its one
            for j in range(1):
                # subscript x[j] if more than one feature
                gradient[j] += x * (y - (sigmoid(np.dot(theta, np.asarray(x)))))
        theta += learning_rate * gradient
    return theta

def test(file, theta, positive_words, negative_words):
    test_features, test_results = read_test_file(file, positive_words, negative_words)
    predictions = []
    actual = []
    # uses trained data to classify test file
    for elem in test_features:
        probability = sigmoid(np.dot(theta,elem))
        if probability > 0.5:
            predictions.append(1)
        else:
            predictions.append(0)
    # gets actual results from test file for accuracy comparisons later on
    for elem in test_results:
        actual.append(elem)
    return predictions, actual

def log_liklihood(theta, features, labels):
    total_sum = 0
    for (x, y) in zip(features, labels):
        # len(x), but we're testing one feature here
        for j in range(1):
            total_sum += (y * (math.log(sigmoid(0)))) +(((1-y) *math.log((1- sigmoid(0)))))
    return total_sum


def main():
    # reads training, postive word, and negative word files
    train_file = open("/Users/yousifmohsen/PycharmProjects/machine_learning/phone_reviews.csv")
    positive_file = open("/Users/yousifmohsen/PycharmProjects/machine_learning/positive.csv")
    negative_file = open("/Users/yousifmohsen/PycharmProjects/machine_learning/negative.csv")
    features, labels, positive_words, negative_words = read_file(train_file, positive_file, negative_file)

    # defines a learning rate for the ML algorithm
    learning_rate = 0.00625
    # trains the data
    theta = train(features, labels, learning_rate)
    # reads through the test file & applied positive and negative binary conversions
    test_file = open("/Users/yousifmohsen/PycharmProjects/machine_learning/utf_phone_reviews.csv")
    result, actual = test(test_file, theta, positive_file, negative_file)
    # compares classified data to actual data in order to see the classification accuracy
    count = 0
    for i in range(len(result)):
        if result[i] == actual[i]:
            count += 1
    # finds the log_liklihood
    total_sum = log_liklihood(theta, features, labels)
    # prints out the accuracy classifications of ML algo
    print(count/len(result))

if __name__ == "__main__":
    main()