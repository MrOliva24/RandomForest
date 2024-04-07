import logging
from abc import ABC, abstractmethod
from logger import mylogger
from Node import Parent, Leaf

logger = mylogger(__name__, logging.CRITICAL)


# We use the design pattern "VISITOR", which allow us to create the different
# functions that we want to implement as classes, and after that, the whole
# tree will visit them both.

# The advantage of this implementation is that, if in the future we want to
# add new functions to the tree, we can just add new classes that inherit the
# abstract "Visitor" one.


class Visitor(ABC):

    # Creation of interface required to apply visitor design pattern

    @abstractmethod
    def visitParent(self, p):
        pass

    @abstractmethod
    def visitLeaf(self, p):
        pass


class FeatureImportance(Visitor):
    def __init__(self):
        self.ocurrences = {}

    def visitParent(self, p: Parent):
        k = p.feature_index
        if k in self.ocurrences.keys():
            self.ocurrences[k] += 1
        else:
            self.ocurrences[k] = 1
        p.left_child.accept_visitor(self)
        p.right_child.accept_visitor(self)

    def visitLeaf(self, leaf: Leaf):
        pass


class PrinterTree(Visitor):
    def __init__(self, depth: int):
        self.depth = depth

    def visitParent(self, parent: Parent):
        print("\t" * self.depth + 'parent, feature index {}, threshold {}'
              .format(parent.feature_index, parent.threshold))

        parent.left_child.accept_visitor(PrinterTree(self.depth + 1))
        parent.right_child.accept_visitor(PrinterTree(self.depth + 1))

    def visitLeaf(self, leaf: Leaf):
        print("\t" * self.depth + 'leaf, label {}'.format(leaf.label))
