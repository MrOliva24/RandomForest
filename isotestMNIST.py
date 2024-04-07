import numpy as np
import pickle
from matplotlib import pyplot as plt
from RandomForest import IsolationForest
def load_MNIST():
    # Method for the MNIST dataset
    with open("mnist.pkl", "rb") as f:
        mnist = pickle.load(f)
    return mnist["training_images"], mnist["training_labels"], mnist["test_images"], mnist["test_labels"]


def test_MNIST(digit):
    X_train, y_train, X_test, y_test = load_MNIST()
    X = np.vstack([X_train, X_test])  # join train and test samples
    y = np.concatenate([y_train, y_test])
    idx_digit = np.where(y == digit)[0]
    X = X[idx_digit]
    downsample = 2  # reduce the number of features = pixels
    X2 = np.reshape(X, (len(X), 28, 28))[:, ::downsample, ::downsample]
    X2 = np.reshape(X2, (len(X2), 28 * 28 // downsample ** 2))
    num_samples = X2.shape[0]
    np.random.seed(123)  # to get replicable results
    idx = np.random.permutation(num_samples)
    X2 = X2[idx]  # shuffle
    y = y[idx]
    iso = IsolationForest(num_trees=2000, ratio_samples=0.5)
    # with multiprocessing=False similar time and results
    iso.fit(X2)
    scores = iso.predict(X2)
    #
    # plt.figure(), plt.hist(scores, bins=100)
    # plt.title('histogram of scores')
    # plt.show()
    percent_anomalies = 0.2
    num_anomalies = int(percent_anomalies * num_samples / 100.)
    idx = np.argsort(scores)
    idx_predicted_anomalies = idx[-num_anomalies:]
    top12 = np.argpartition(scores, -12)[-12:]
    for i in range(2):
        for j in range(6):
            n_sample = 6 * i + j
            plt.subplot(2, 6, n_sample + 1)
            plt.imshow(np.reshape(X2[top12[n_sample]], (14, 14)),
                       interpolation='nearest', cmap=plt.cm.gray)
            plt.axis('off')
    plt.show()

test_MNIST(9)
