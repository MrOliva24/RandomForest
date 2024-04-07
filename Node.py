import logging
from abc import ABC, abstractmethod
from logger import mylogger
import numpy as np
logger = mylogger(__name__, logging.ERROR)


class Node(ABC):
    @abstractmethod
    def predict(self, x):
        pass

    @abstractmethod
    def accept_visitor(self, visitor):
        # Method required to accept the visitor
        pass


class Leaf(Node):

    def __init__(self, label: int):
        self.label = label

    def predict(self, x: np.ndarray[float]) -> int:
        return self.label

    def accept_visitor(self, visitor):
        visitor.visitLeaf(self)
        logger.info("\nVisited the leaf {}".format(self))


class Parent(Node):

    def __init__(self, feature_index: int, threshold: float):

        self.feature_index = feature_index
        self.threshold = threshold
        self.left_child = None
        self.right_child = None

    def predict(self, x: np.ndarray[float]) -> np.ndarray[float]:

        if x[self.feature_index] < self.threshold:
            logger.info("\nPrediction made on the parent {}".format(self))
            return self.left_child.predict(x)
        else:
            logger.info("\nPrediction made on the parent {}".format(self))
            return self.right_child.predict(x)

    def accept_visitor(self, visitor):
        # For the type hints we should say that visitor is from class Visitor
        # but if we do so, it appears a problem of circular importation with
        # Visitor file, so we can
        visitor.visitParent(self)
