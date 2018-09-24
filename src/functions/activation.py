# libs
import numpy


def sigmoid(output):
    #code from toward datascience
    """return the sigmoid of output"""
    return 1 / (1 + numpy.exp(-output))
