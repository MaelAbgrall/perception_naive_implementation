# libs
import numpy


def sigmoid(output):
    """return the sigmoid of output"""
    return 1 / (1 + numpy.exp(-output))
