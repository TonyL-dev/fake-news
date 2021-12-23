#Anthony Likhachov
#1005124839
#hw1
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import export_graphviz
from sklearn.metrics import accuracy_score
import pydotplus
from sklearn.externals.six import StringIO
from IPython.display import Image
import pandas as pd
import math

def load_data():
    # create data
    clean_fake = np.array([])
    clean_real = np.array([])
    with open("C:/Users/coolg/PycharmProjects/pythonProject/Assignment 1/Data/clean_fake.txt") as f:
        clean_fake = f.read().splitlines()

    label_fake = np.full((len(clean_fake), 1), "FAKE")

    with open("C:/Users/coolg/PycharmProjects/pythonProject/Assignment 1/Data/clean_real.txt") as f:
        clean_real = f.read().splitlines()

    label_real = np.full((len(clean_real), 1), "REAL")

    dataY = np.concatenate((label_fake, label_real), axis=0)
    dataX = np.concatenate((clean_fake, clean_real), axis=0)

    # split data
    train_ratio = 0.7
    validation_ratio = 0.15
    test_ratio = 0.15

    #set a random state for consistency purposes
    x_train, x_test, y_train, y_test = train_test_split(dataX, dataY, test_size=1 - train_ratio, random_state=0)

    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test,
                                                    test_size=test_ratio / (test_ratio + validation_ratio),
                                                    random_state=0)

    # vectorize
    vectorizer = TfidfVectorizer()
    tfidf_train = vectorizer.fit_transform(x_train)
    tfidf_val = vectorizer.transform(x_val)
    tfidf_test = vectorizer.transform(x_test)

    #return so other functions can use the vectorized datasets
    return tfidf_train, tfidf_val, tfidf_test, y_train, y_val, y_test, vectorizer


def select_model():
    tfidf_train, tfidf_val, tfidf_test, y_train, y_val, y_test, vectorizer = load_data()
    criteria = ["gini", "entropy"]
    max_depth_range = list(range(1, 6))

    for depth in max_depth_range:
        for split in criteria:
            clf = tree.DecisionTreeClassifier(max_depth=depth, criterion=split)
            clf.fit(tfidf_train, y_train)

            score = clf.score(tfidf_val, y_val)
            print("With a depth of {depth} and using a {split} split, the accuracy is {score}".format(depth=depth,
                                                                                                      split=split,
                                                                                                      score=score))

    # code to output as a png and on screen
    # dot_data = StringIO()
    # export_graphviz(clf, out_file=dot_data,
    #               filled=True, rounded=True,
    #               feature_names = vectorizer.get_feature_names()
    #               )
    # graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    # graph.write_png('fake_news.png')
    # Image(graph.create_png())
    # tree.plot_tree(clf)

#computes Information Gain given a word and a TF-IDF value
def compute_information_gain(word, tfIDF):
    dataLessX, dataLessY, dataMoreX, dataMoreY = [], [], [], []
    tfidf_train, tfidf_val, tfidf_test, y_train, y_val, y_test, vectorizer = load_data()
    index = vectorizer.get_feature_names().index(word)

    # split data into 2 sets
    for i in range(tfidf_train.shape[0]):
        if (tfidf_train[i][(0, index)] <= tfIDF):
            dataLessX.append(tfidf_train[i][(0, index)])
            dataLessY.append(y_train[i][0])
        else:
            dataMoreX.append(tfidf_train[i][(0, index)])
            dataMoreY.append(y_train[i][0])
    dependent_values = np.unique(y_train)

    # calculate parent entropy
    parent_entropy = 0
    for value in dependent_values:
        p = calculate_p(y_train, value)
        parent_entropy -= p * math.log(p, 2)

    # calculate child entropy
    entropy_below = 0
    entropy_above = 0

    # calculate entropy for both splits
    for value in dependent_values:
        p = calculate_p(dataLessY, value)
        entropy_below -= p * math.log(p, 2)

        p = calculate_p(dataMoreY, value)
        entropy_above -= p * math.log(p, 2)
    # weight the entropy by number of data points in each split
    weighted_entropy = len(dataLessY) / len(y_train) * entropy_below
    weighted_entropy += len(dataMoreY) / len(y_train) * entropy_above

    return parent_entropy - weighted_entropy


def calculate_p(arr, value):
    count = 0
    for element in arr:
        if element == value:
            count += 1
    return count / len(arr)


# load_data()
select_model()
#compute several information gain given the photo output
print("Using the word trump with a TF-IDF of 0.055, the Information Gain is {ig}".format(ig = compute_information_gain("trump", 0.055)))
print("Using the word trumps with a TF-IDF of 0.164, the Information Gain is {ig}".format(ig = compute_information_gain("trumps", 0.164)))
print("Using the word coal with a TF-IDF of 0.215, the Information Gain is {ig}".format(ig = compute_information_gain("coal", 0.215)))