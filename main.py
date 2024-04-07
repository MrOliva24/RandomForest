import logging
import os
import pickle

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris

from ImpurityMeasure import *
from RandomForest import *
from logger import mylogger

logger = mylogger(__name__, logging.CRITICAL)
logger.disabled = True

iris = load_iris()  # it's a dictionary

'''
We have created this file in order to have an easy executable 
so we can use any dataset by introducing an input and then 
decide whether we want to use the regressor or the classifier 
and if we want to add the multiprocessing or the extra-trees
'''


def execRF(X_train, y_train, X_test, y_test, num_features, opt):
    t0 = time.time()
    max_depth = 20  # maximum number of levels of a decision tree
    min_size_split = 20  # if less, do not split a node
    ratio_samples = 0.7  # sampling with replacement
    num_trees = 80  # number of decision treesº
    num_random_features = int(np.sqrt(num_features))
    # number of features to consider at
    # each node when looking for the best split
    criterion = Gini()  # 'gini' or 'entropy'
    logger.info("\nCriterion {}".format(criterion))
    rf = RandomForestClassifier(max_depth, min_size_split, ratio_samples,
                                num_trees, num_random_features, criterion, opt)
    # train = make the decision trees
    rf.fit(X_train, y_train)
    # classification
    ypred = rf.predict(X_test)

    logger.info("\nPrediction test {}".format(ypred))
    # compute accuracy
    num_samples_test = len(y_test)
    num_correct_predictions = np.sum(ypred == y_test)
    logger.info("\nCorrect predictions {}".format(num_correct_predictions))
    accuracy = num_correct_predictions / float(num_samples_test)
    print('accuracy {} %'.format(100 * np.round(accuracy, decimals=2)))
    tf = time.time()
    print("\nTotal time : {} seconds\n".format(tf - t0))
    return rf


'''
We prepare different methods in order to load the different datasets
'''


def load_daily_min_temperatures():
    # Method for the temperatures dataset

    df = pd.read_csv('https://raw.githubusercontent.com/jbrownlee/'
                     'Datasets/master/daily-min-temperatures.csv')
    # Minimum Daily Temperatures Dataset over 10 years (1981-1990)
    # in Melbourne, Australia. The units are in degrees Celsius.
    # These are the features to regress:
    day = pd.DatetimeIndex(df.Date).day.to_numpy()  # 1...31
    month = pd.DatetimeIndex(df.Date).month.to_numpy()  # 1...12
    year = pd.DatetimeIndex(df.Date).year.to_numpy()  # 1981...1999
    X = np.vstack([day, month, year]).T  # np array of 3 columns
    y = df.Temp.to_numpy()
    return X, y


def load_sonar():
    # Method for the sonar dataset

    df = pd.read_csv('sonar.all-data', header=None)
    X = df[df.columns[:-1]].to_numpy()
    y = df[df.columns[-1]].to_numpy(dtype=str)
    y = (y == 'M').astype(int)  # M = mine, R = rock
    return X, y


def load_MNIST():
    # Method for the MNIST dataset

    with open("mnist.pkl", "rb") as f:
        mnist = pickle.load(f)
    return mnist["training_images"], mnist["training_labels"], mnist["test_images"], mnist["test_labels"]


def test_regression(opt, last_years_test=2):
    # Method that will be used for the regressor random forest

    X, y = load_daily_min_temperatures()
    plt.figure()
    plt.plot(y, '.-')
    plt.xlabel('day in 10 years'), plt.ylabel('min. daily temperature')
    idx = last_years_test * 365
    Xtrain = X[:-idx, :]  # first years
    Xtest = X[-idx:]
    ytrain = y[:-idx]  # last years
    ytest = y[-idx:]
    max_depth = 10
    min_size_split = 5
    ratio_samples = 0.5
    num_trees = 50
    num_random_features = 2
    criterion = SSE()

    RFR = RandomForestRegressor(max_depth, min_size_split, ratio_samples, num_trees, num_random_features, criterion,
                                opt)

    RFR.fit(Xtrain, ytrain)
    ypred = RFR.predict(Xtest)

    plt.figure()
    x = range(idx)
    for t, y1, y2 in zip(x, ytest, ypred):
        plt.plot([t, t], [y1, y2], 'k-')
    plt.plot([x[0], x[0]], [ytest[0], ypred[0]], 'k-', label='error')
    plt.plot(x, ytest, 'g.', label='test')
    plt.plot(x, ypred, 'y.', label='prediction')
    plt.xlabel('day in last {} years'.format(last_years_test))
    plt.ylabel('min. daily temperature')
    plt.legend()
    errors = ytest - ypred
    rmse = np.sqrt(np.mean(errors ** 2))
    plt.title('root mean square error : {:.3f}'.format(rmse))
    plt.show()


def test_classifier(opt):
    # Method that will be used for the classifier random forest

    answer = None
    while answer not in ["sonar", "iris", "temperatures", "MNIST"]:
        print("Select dataset to work with:\n  -iris\n  -sonar\n  -temperatures\n  -MNIST")
        answer = input()
    if answer != "MNIST":
        if answer == "iris":
            logger.info("\nIris dataset selected")
            iris = load_iris()
            X, y = iris.data, iris.target
        elif answer == "sonar":
            logger.info("\nSonar dataset selected")
            X, y = load_sonar()
        elif answer == "temperatures":
            logger.info("\nTemperatures dataset selected")
            X, y = load_daily_min_temperatures()

        ratio_train = 0.7
        ratio_test = 0.3

        num_samples, num_features = X.shape
        idx = np.random.permutation(range(num_samples))

        num_samples_train = int(num_samples * ratio_train)
        num_samples_test = int(num_samples * ratio_test)
        idx_train = idx[:num_samples_train]
        idx_test = idx[num_samples_train: num_samples_train + num_samples_test]
        X_train, y_train = X[idx_train], y[idx_train]
        X_test, y_test = X[idx_test], y[idx_test]

    else:
        logger.info("\nMNIST dataset selected")
        X_train, y_train, X_test, y_test = load_MNIST()

    if answer == "iris":
        rf = execRF(X_train, y_train, X_test, y_test, num_features, opt)
        occurrences = rf.feature_importance()
        print('Iris occurrences for {} trees'.format(rf.num_trees))
        print(occurrences)
        rf.print_trees()
    elif answer == "sonar":
        rf = execRF(X_train, y_train, X_test, y_test, num_features, opt)
        occurrences = rf.feature_importance()  # a dictionary
        counts = np.array(list(occurrences.items()))
        plt.figure()
        plt.bar(counts[:, 0], counts[:, 1])
        plt.xlabel('feature')
        plt.ylabel('occurrences')
        plt.title('Sonar feature importance\n{} trees'.format(rf.num_trees))
        plt.show()
        logger.info("\nPlotted the sonar dataset")
        rf.print_trees()
    elif answer == "MNIST":
        if not os.path.exists('rf_mnist.pkl'):
            rf = execRF(X_train, y_train, X_test, y_test, int(np.sqrt(X_train.shape[1])), opt)
            with open('rf_mnist.pkl', 'wb') as f:
                pickle.dump(rf, f)

        else:
            with open('rf_mnist.pkl', 'rb') as f:
                rf = pickle.load(f)
        occurrences = rf.feature_importance()
        ima = np.zeros(28 * 28)
        for k in occurrences.keys():
            ima[k] = occurrences[k]
        plt.figure()
        plt.imshow(np.reshape(ima, (28, 28)))
        plt.colorbar()
        plt.title('Feature importance MNIST\n {} trees, {}% samples/tree'.format(rf.num_trees,
                                                                                 round(1 - rf.ratio_samples, 2) * 100))
        plt.show()
        logger.info("\nPlotted MNIST dataset")


if __name__ == "__main__":
    print("Select Random forest method:\n  0.-Regressor\n  1.-Classifier")
    method = int(input())
    print("\nSelect optimization:\n  0.-None\n  1.-Extra trees\n  2.-Multiprocessing")
    optimization = int(input())
    if method == 0:
        test_regression(optimization)
    else:
        test_classifier(optimization)
