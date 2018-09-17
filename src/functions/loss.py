# integrated
import numpy

def mean_squared(y_truth, y_predict):
    loss = (1/2) * ((y_truth - y_predict) * (y_truth - y_predict))
    return loss


def absolute(y_truth, y_predict):
    loss = abs(y_truth - y_predict)
    return loss


def huber(y_truth, y_predict, d_part):
    absolute = abs(y_truth - y_predict)

    if(absolute <= d_part):
        loss = mean_squared(y_truth, y_predict)
    if(absolute > d_part):
        loss = (d_part * absolute) - (1/2)*(d_part * d_part)

    return loss

def cross_enthropy(y_truth, y_predict):
    loss = - (y_truth * numpy.log(y_predict) + ((1 - y_truth) * numpy.log(1 - y_predict)))
    return loss
