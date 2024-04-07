import multiprocessing
import time
import logging
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict
from Dataset import Dataset
from Visitor import PrinterTree, FeatureImportance
from Node import Node, Parent, Leaf
from ImpurityMeasure import Impurity
from logger import mylogger

logger = mylogger(__name__, logging.CRITICAL)

"""During the whole project we will follow the PEP8 style, for more 
information use the next link: https://pep8.org/"""


class RandomForest(ABC):
    """
    Random Forest class, to represent the trees

    Attributes
    ------------
    num_trees : int number of trees we want to work with from the random forest

    max_depth : int the maximum depth of the tree.

    min_size : int minimum size of terminal nodes.

    ratio_samples : float ratio of data we take for each workout
    num_random_features : int number of features to consider at each node
    when looking for the best split

    criterion : Impurity the function to measure the quality of a split.

    Methods
    -------
    fit (X,y) : will fit the model to the input training instances

    _combine_predictions(predictions): abstract method, according to the type
    of random forest we are applying, it realizes a type of combination

    predict(X): will perform predictions on the testing instances

    _make_decision_trees(dataset) : creates the list of decision trees
    applying the criteria

    _make_node(dataset, depth) : creates the nodes of the trees,
    and classifies if are leaf or we still do not know if are parent or leaf

    _make_parent_or_leaf(dataset, depth) : calculates if the node entering is
    a parent or a leaf, and classifies that

    _make_leaf(dataset) : abstract method, that according to the type of
    random forest we are applying the leaf will create with appropriate

    method calculate_impurity(dataset) : calculus of the impurity selected

    _CART_cost(left_dataset, right_dataset) : minimizes the function cost
    feature_importance(num_features) : decides which feature to choose and
    the threshold to apply to it in order to follow the left or right branch.
    A simple way to derive importance is to take all the nodes from all the
    trees and count how many times each feature is used.

    print_trees: print, and show the trees created in the random forest

    """

    def __init__(self, max_depth: int, min_size_split: int,
                 ratio_samples: float,
                 num_trees: int, num_random_features: int, criterion: Impurity,
                 optimization: bool):

        """

        Parameters
        ----------
        num_trees : int number of trees we want to work with from the random
        forest

        max_depth : int the maximum depth of the tree.

        min_size_split : int minimum size of terminal nodes.

        ratio_samples : float ratio of data we take for each workout

        num_random_features : int number of features to consider at each node
        when looking for the best split

        criterion : Impurity the function to measure the quality of a split.

        optimization: string decides which optimization feature to use
        """

        self.num_trees = num_trees
        self.min_size = min_size_split
        self.max_depth = max_depth
        self.ratio_samples = ratio_samples
        self.criterion = criterion
        self.num_random_features = num_random_features
        self.optimization = optimization

    def fit(self, X: np.ndarray[float], y: np.ndarray[int]):
        # PEP-8 problems
        """
        fits the model to the input training instances

         Parameters
         ----------
         X : numpy array
             dataset part X
         y : numpy array
             dataset part y
         """
        if self.optimization == 2:
            self._make_decision_trees_multiprocessing(Dataset(X, y))
        else:
            self._make_decision_trees(Dataset(X, y))

    @abstractmethod
    def _combine_predictions(self, predictions: List):
        """
        Abstract method, will be coded in each subclass

        Parameters
        ----------
        predictions : numpy array [float]
            array of the predictions, calculated in predict
        """
        pass

    def _make_decision_trees(self, dataset: Dataset):
        """
        Creates the list of decision trees applying the criteria
        Parameters
        ----------
        dataset : Dataset
            array of the dataset we are working with
        """
        self.decision_trees = []
        for i in range(self.num_trees):
            subset = dataset.random_sampling(self.ratio_samples)
            tree = self._make_node(subset, 1)  # The root of the decision tree
            self.decision_trees.append(tree)

    def _make_node(self, dataset: Dataset, depth: int) -> Node:
        """
        Creates the nodes of the trees, and classifies if are leaf,
        or we still do not know if are parent or leaf
        Parameters
        ----------
        dataset : Dataset
            array of the dataset we are working with
        depth : int
            node depth

        Returns
        -------
        node : Node
            returns the node that we just create

        """
        if depth == self.max_depth \
                or dataset.num_samples <= self.min_size \
                or len(np.unique(dataset.y)) == 1:
            # Last condition is true if all samples belong to the same class
            node = self._make_leaf(dataset)
        else:
            node = self._make_parent_or_leaf(dataset, depth)
        return node

    @abstractmethod
    def _make_leaf(self, dataset: Dataset, depth=None):
        """
        Abstract method will be coded in each subclass

        Parameters
        ----------
        dataset : numpy array [float]
            array of the dataset we are working with
        """

        pass

    def _best_split(self, idx_features: np.ndarray[int], dataset: Dataset) -> \
            Tuple[int, float, float, List[Dataset]]:

        """
        Find the best pair (feature, threshold) by exploring all possible pairs

        Parameters
        ------------
        idx_features: numpy array [int]

        dataset: Dataset

        Returns
        ------------
        best_feature_index: int

        best_threshold: float

        minimum_cost: float

        best_split: List [dataset]
        """
        best_feature_index, best_threshold, minimum_cost, best_split = \
            np.Inf, np.Inf, np.Inf, None
        if self.optimization == 1:
            for idx in idx_features:
                min_val = dataset.X[:, idx].min()
                max_val = dataset.X[:, idx].max()
                random = (min_val + (max_val - min_val)) * np.random.rand()
                left_dataset, right_dataset = dataset.split(idx, random)
                cost = self._CART_cost(left_dataset, right_dataset)
                # J(k,v)
                if cost < minimum_cost:
                    best_feature_index, best_threshold, minimum_cost, \
                        best_split = idx, random, cost, [left_dataset,
                                                         right_dataset]
        else:
            for idx in idx_features:
                values = np.unique(dataset.X[:, idx])
                for val in values:
                    left_dataset, right_dataset = dataset.split(idx, val)
                    cost = self._CART_cost(left_dataset,
                                           right_dataset)  # J(k,v)
                    if cost < minimum_cost:
                        best_feature_index, best_threshold, minimum_cost, \
                            best_split = idx, val, cost, [left_dataset,
                                                          right_dataset]
        print(best_feature_index, best_threshold, minimum_cost, best_split)
        return best_feature_index, best_threshold, minimum_cost, best_split

    def _make_parent_or_leaf(self, dataset: Dataset, depth: int) -> Node:
        """
        Calculates if the node entering is a parent or a leaf, and classifies


        Parameters
        ----------
        dataset : Dataset
            Array containing the dataset
        depth : int
           node depth

        Returns
        -------
        node : Node
            returns the node that we just create
        """
        # select a random subset of features, to make trees more diverse
        idx_features = np.random.choice(range(dataset.num_features),
                                        self.num_random_features, replace=False)
        best_feature_index, best_threshold, minimum_cost, best_split = \
            self._best_split(idx_features, dataset)
        left_dataset, right_dataset = best_split
        assert left_dataset.num_samples > 0 or right_dataset.num_samples > 0
        if left_dataset.num_samples == 0 or right_dataset.num_samples == 0:
            '''this is a special case : dataset has samples of at least two 
            classes but the best split is moving all samples to the left or 
            right dataset and none to the other, so we make a leaf instead of 
            a parent'''
            return self._make_leaf(dataset, depth)
        else:
            node = Parent(best_feature_index, best_threshold)
            node.left_child = self._make_node(left_dataset, depth + 1)
            node.right_child = self._make_node(right_dataset, depth + 1)
            return node

    def _CART_cost(self, left_dataset: Dataset,
                   right_dataset: Dataset) -> float:
        """
        Minimizes function cost


        Parameters
        ------------
            left_dataset : Dataset
            right_dataset : Dataset

        Returns
        ------------
            returns the value of the cart cost
        """
        left_cost = left_dataset.num_samples / (
                left_dataset.num_samples +
                right_dataset.num_samples) * self._impurity(left_dataset)

        right_cost = right_dataset.num_samples / (
                left_dataset.num_samples +
                right_dataset.num_samples) * self._impurity(right_dataset)
        return left_cost + right_cost

    def _impurity(self, dataset: Dataset) -> float:
        """
        Computes the impurity of the dataset with the selected method

        Parameters
        ------------
            dataset : numpy array [float]
                Array containing the dataset

        Returns
        ------------
            returns the computed dataset with the criterion wanted
        """
        return self.criterion.compute(dataset)

    def predict(self, X: np.ndarray[float]) -> np.ndarray[float]:
        """
        Performs predictions on the testing instances

        Parameters
        ------------
            X : numpy ndarray
                X subdivision of Dataset

        Returns
        ------------
            np.array(y_pred) : numpy ndarray
                returns dataset prediction
        """

        y_pred = []
        for x in X:
            predictions = [tree.predict(x) for tree in self.decision_trees]
            y_pred.append(self._combine_predictions(predictions))
        return np.array(y_pred)

    def feature_importance(self) -> Dict:
        """
        Mesures how important is each of the features counting how many times
        each feature is used

        Returns
        ------------
            feat_imp_visitor.occurrences : dictionary
                returns number of times each feature has been used
        """
        feat_imp_visitor = FeatureImportance()
        for tree in self.decision_trees:
            tree.accept_visitor(feat_imp_visitor)
        return feat_imp_visitor.ocurrences

    def print_trees(self):
        # Prints the trees used for the Random Forest
        for tree in self.decision_trees:
            tree_printer = PrinterTree(0)
            tree.accept_visitor(tree_printer)

    # --------------------------------------------------
    #   MULTIPROCESSING
    # --------------------------------------------------

    def _target(self, dataset: Dataset, nproc: int) -> Node:
        """
        Prints the process that starts each time

        Parameters
        -------------
            dataset : numpy array [float]
                Array containing the dataset

            nproc : int
                the number of each process
        Returns
        -------------
            tree : Node
                returns the tree created from a random sampling
        """
        subset = dataset.random_sampling(self.ratio_samples)
        tree = self._make_node(subset, 1)
        print('process {} ends'.format(nproc))
        return tree

    def _make_decision_trees_multiprocessing(self, dataset: Dataset):
        """
        Prints the average number of seconds that were needed for each tree

        Parameters
        -------------
            dataset : numpy array [float]
                Array containing the dataset
       """
        t1 = time.time()
        with multiprocessing.Pool() as pool:
            self.decision_trees = pool.starmap(self._target,
                                               [(dataset, nprocess) for nprocess
                                                in range(self.num_trees)])
            # use pool.map instead if only one argument for _target
        t2 = time.time()
        print('{} seconds per tree'.format((t2 - t1) / self.num_trees))


class RandomForestClassifier(RandomForest):
    """
    Random forest class using its classifier feature

    Inheritance
    -----------
    RandomForest

    Methods
    -----------
    _combine_predictions(predictions):
        combination of predictions based on the classifier

    _make_leaf(dataset):
        creates a Leaf object based on the most frequent label of the dataset
    """

    def _combine_predictions(self, predictions: np.ndarray[float]) \
            -> np.ndarray[int]:
        """
        Parameters
        ----------
            predictions : numpy array [float]
                array of the predictions, calculated in predict

        Returns
        -------
            index: numpy array [int]
        """
        return np.argmax(np.bincount(predictions))

    def _make_leaf(self, dataset: Dataset, depth=None) -> Leaf:
        """
        Parameters
        ----------
            dataset : numpy array
                Array containing the dataset

        Returns
        -------
            Leaf : Leaf
                leaf created about the most_frequent_label
        """
        return Leaf(int(dataset.most_frequent_label()))


class RandomForestRegressor(RandomForest):
    """
    Random forest class applying using its regression feature

    Inheritance
    -----------
    RandomForest

    Methods
    -------
     _combine_predictions(predictions):
        combination of predictions based in the regressor

    _make_leaf(dataset):
        Creates Leaf object based on the y subdivision of the dataset

    """

    def _combine_predictions(self, predictions: np.ndarray[float]) -> \
            np.ndarray[float]:
        """
        Average of predicted values for an X

        Parameters
        ----------
            predictions : numpy array [float]
                Array containing the dataset

        Returns
        -------
            np.mean(predictions) : float
                returns te mean value of predictions
        """
        return np.mean(predictions)

    def _make_leaf(self, dataset: Dataset, depth=None) -> Leaf:
        """
        Parameters
        ------------
            dataset : numpy array [float]
                Array containing the dataset

        Returns
        ------------
            Leaf(dataset.mean_value())
                A Leaf node created from the mean value of the y numpy array

        """
        return Leaf(int(dataset.mean_value()))


class IsolationForest(RandomForest):
    """
    Random forest class applying using its anomaly detection feature

    Inheritance
    -----------
    RandomForest

    Methods
    -------
     _combine_predictions(predictions):
        combination of predictions based in the regressor

    _make_leaf(dataset, depth):
        Creates Leaf object based on the y subdivision of the dataset

    _make_node(dataset, depth):
        Decides to create Parent or Leaf depending on depth

    fit(X):
        will fit the model to the input training instances

    predict(X):
        will perform predictions on the testing instances

    _best_split(idx_features, dataset):
        Find the best pair (feature, threshold) by exploring all possible pairs
    """

    def __init__(self, ratio_samples: float, num_trees: int):
        """
        Partly inherits the init method from the abstract class

        Parameters
        ----------
        num_trees : int
            number of trees we want to work with from the random forest
        ratio_samples : float
            ratio of data we take for each workout
        """
        super().__init__(None, 1, ratio_samples, num_trees, 1, None, None)
        # TypeHint errors because of IsolationForest initialization
        self.test_size = None
        self.train_size = None

    def _make_leaf(self, dataset: Dataset, depth: int) -> Leaf:
        # Signature error because of override of method
        """
        Creates leafs from the depth parameter

        Parameters
        ----------
        dataset : Dataset
            array of the dataset we are working with
        depth : int
            the number of nodes created since the beginning until this one

        Returns
        ----------
            Leaf(depth)

        """

        return Leaf(depth)

    def _make_node(self, dataset: Dataset, depth: int) -> Node:
        """
        Creates nodes

        Parameters
        ----------
        dataset : Dataset
            array of the dataset we are working with
        depth : int
            the number of nodes created since the beginning until this one

        Returns
        ----------
            node
        """
        if depth == self.max_depth or dataset.num_samples <= self.min_size:
            node = self._make_leaf(dataset, depth)
            logger.debug("Leaf Node created")
        else:
            node = self._make_parent_or_leaf(dataset, depth)
            logger.debug("Parent or Leaf Node created")
        return node

    def fit(self, X: np.ndarray[float]):
        # Signature error because of override of method
        """
        fits the model to the input training instances

        Parameters
        ----------
        X : numpy array
            dataset part X

        Returns
        ----------
        Decision tree
            Created from the Dataset part X
        """
        self.max_depth = np.log2(len(X))
        self.train_size = len(X)
        y = np.array([0] * self.train_size)
        return self._make_decision_trees(Dataset(X, y))

    def predict(self, X: np.ndarray[float]) -> np.ndarray[float]:
        """
        Performs predictions on the testing instances

        Inherits the predict method from the abstract class

        Parameters
        ------------
            X : numpy ndarray [float]
                X subdivision of Dataset

        Returns
        ------------
            np.array(y_pred) : numpy ndarray [float]
                returns dataset prediction
        """
        self.test_size = len(X)
        return super().predict(X)

    def _combine_predictions(self, predictions: np.ndarray[float]) -> int:
        """
        Parameters
        ----------
        predictions : numpy array [float]
            array of the predictions, calculated in predict

        Returns ---------- Score : float value that determines the
        probability of being an anomaly, the value is always between 0 and 1,
        if it's near 0, is probably a normal value, if it's near 1, it might
        be an anomaly
        """
        ehx = np.mean(predictions)
        cn = 2 * (np.log(self.train_size - 1) + 0.57721) - 2 * (
                self.train_size - 1) / float(self.test_size)
        return 2 ** (-ehx / cn)

    def _best_split(self, idx_features: np.ndarray[int], dataset: Dataset) -> \
            Tuple[int, float, None, List[Dataset]]:
        """
        Find the best pair (feature, threshold) by exploring all possible pairs

        Parameters
        ----------
        idx_features : numpy array [int]

        dataset : Dataset
            array of the dataset we are working with

        Returns
        ----------
        k : int
            random value of the idx_features
        value : float
            value for the split
        """
        k = np.random.choice(idx_features)
        min_val = dataset.X[:, k].min()
        max_val = dataset.X[:, k].max()
        multiplier = np.random.random()  # This code line does a random value
        # between 0 and 1
        value = multiplier * (max_val - min_val) + min_val
        left_dat, right_dat = dataset.split(k, value)
        dataset_split = [left_dat, right_dat]
        return k, value, None, dataset_split
