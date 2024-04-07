import matplotlib.pyplot as plt
from RandomForest import IsolationForest
import numpy as np
import pandas as pd


def test_credit_card_fraud():
    df = pd.read_csv('creditcard_10K.csv'
    , header=None)
    X = np.array(df)
    X = X[:,1:] # remove first feature
    y = X[:,-1]
    X = X[:,:-1]
    del(df)
    num_samples = len(X)
    print('{} number of samples'.format(num_samples))
    np.random.seed(123) # to get replicable results
    idx = np.random.permutation(num_samples)
    X = X[idx] # shuffle
    y = y[idx]
    print('{} samples, {} outliers, {} % '.format(len(y), y.sum(),
        np.round(100*y.sum()/len(y),decimals=3)))
    iso = IsolationForest(num_trees=500, ratio_samples=0.1)
    # with multiprocessing=False similar time and results
    iso.fit(X)
    scores = iso.predict(X)
    plt.figure(), plt.hist(scores, bins=100)
    plt.title('histogram of scores')
    percent_anomalies = 0.5
    num_anomalies = int(percent_anomalies * num_samples / 100.)
    idx = np.argsort(scores)
    idx_predicted_anomalies = idx[-num_anomalies:]
    precision = y[idx_predicted_anomalies].sum() / num_anomalies
    print('precision for {} % anomalies : {} %' \
          .format(percent_anomalies, 100 * precision))
    recall = y[idx_predicted_anomalies].sum() / y.sum()
    print('recall for {} % anomalies : {} %' \
          .format(percent_anomalies, 100 * recall))
    plt.show()

test_credit_card_fraud()