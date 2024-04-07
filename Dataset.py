import logging

import numpy as np

from logger import mylogger

logger = mylogger(__name__, logging.ERROR)


class Dataset:

    def __init__(self, X: np.ndarray[float], y: np.ndarray[int]):
        self.X = np.array(X)
        self.y = y
        self.num_samples = len(y)
        self.num_features = X.shape[1]

    def random_sampling(self, ratio_samples: float) -> "Dataset":
        # Generates a random smaller dataset proportional to the ratio
        # specified in the hyperparameters
        assert (0 < ratio_samples < 1), "Ratio must be between 0 and 1"
        index = np.random.randint(0, self.num_samples,
                                  int(ratio_samples * self.num_samples))
        return Dataset(self.X[index], self.y[index])

    def most_frequent_label(self) -> np.ndarray[int]:
        # Returns most frequent class on the dataset by analyzing a numpy
        # array, so it ends up returning: The most frequent argument in an
        # array composed by the counts of each element repetition
        logger.info("\n Most frequent label: {}".format(np.argmax(
            np.bincount(self.y))))
        return np.argmax(np.bincount(self.y))

    def distribution(self) -> float:
        return np.bincount(self.y) / np.sum(np.bincount(self.y))
        # Returns "percentage" of each label in dataset

    def split(self, idx, val) -> tuple["Dataset", "Dataset"]:
        index_left = self.X[:, idx] < val
        index_right = self.X[:, idx] >= val
        logger.info("\n Value used in order to make a split on the Dataset: {}".
                    format(val))
        return Dataset(self.X[index_left], self.y[index_left]), \
            Dataset(self.X[index_right], self.y[index_right])
        # Divides dataset into 2 subsets

    def mean_value(self) -> np.ndarray[float]:
        return np.mean(self.y)
