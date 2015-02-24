from abc import ABCMeta
from abc import abstractmethod


class AbstractDetector(object):
    """
    Base class from which detectors should extend
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def find_rectangles(self):
        """
        Find positions of the license plates in the image

        :return: Array of rectangles represented as points
        """
        pass

    @abstractmethod
    def _check_sizes(self, candidate):
        """
        Perform size check on the specified rectangle
        :param candidate: Rectangle on which to perform the check
        :return: True if conditions satisfied, otherwise False
        """
        pass