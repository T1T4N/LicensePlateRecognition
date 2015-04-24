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
    def _check_size(self, candidate, area=-1):
        """
        Perform size check on the specified rectangle

        :param candidate: Rectangle on which to perform the check
        :param area: Optional if area is given as an argument
        :return: True if conditions satisfied, otherwise False
        """
        pass
