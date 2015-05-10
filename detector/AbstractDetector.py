from abc import ABCMeta
from abc import abstractmethod


class AbstractDetector(object):
    """
    Base class from which detectors should extend
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def find_plates(self):
        """
        Find the license plates in the image

        :rtype: list[(numpy.array, numpy.array)]
        :return: List of tuples containing the plate image and the plate rectangle location
            The plates returned must be a grayscale image with black background and white characters
        """
        pass

    @abstractmethod
    def _check_size(self, candidate, area=None):
        """
        Perform size check on the specified rectangle

        :param candidate: Rectangle on which to perform the check
        :type area: float | None
        :param area: Optional area specified
        :rtype: bool
        :return: True if conditions satisfied, otherwise False
        """
        pass
