import numpy as np
import logging
from abc import ABC, abstractmethod
from logger import mylogger
from Dataset import Dataset

logger = mylogger(__name__, logging.ERROR)


class Impurity(ABC):

    # Use of an abstract class as an interface to create a template to
    # instance the different impurity measures

    @abstractmethod
    def compute(self, dataset):
        pass


# All the following classes inherit the abstract class, so they only have to
# add the specific code for their own "compute" method, which will be
# different for each one

class Gini(Impurity):

    def compute(self, dataset: Dataset) -> float:
        logger.debug("\nUsed dataset distribution {}".
                     format(dataset.distribution()))
        return 1 - np.sum(dataset.distribution() ** 2)


class Entropy(Impurity):

    def compute(self, dataset: Dataset) -> float:
        distro = dataset.distribution()
        distro = distro[distro > 0]  # We get rid of the values equal to 0
        # because we are using logarithms
        logger.debug("\nUsed dataset distribution {}".format(distro))
        return - np.sum(distro * np.log(distro))


class SSE(Impurity):

    def compute(self, dataset: Dataset) -> np.ndarray[float]:
        ymitj = (sum(dataset.y)) / (dataset.X.shape[1])
        sse = np.sum((dataset.y - ymitj) ** 2)
        return sse
