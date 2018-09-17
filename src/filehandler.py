# libs
import numpy
import h5py


def import_dataset(path):
    train_dataset = h5py.File(path + 'trainCats.h5', "r")
    x_train = numpy.array(train_dataset["train_set_x"][:])
    y_train = numpy.array(train_dataset["train_set_y"][:])
    y_train = y_train.reshape((1, y_train.shape[0]))

    test_dataset = h5py.File(path + 'testCats.h5', "r")
    x_validation = numpy.array(test_dataset["test_set_x"][:])
    y_validation = numpy.array(test_dataset["test_set_y"][:])
    y_validation = y_validation.reshape((1, y_validation.shape[0]))

    classes = numpy.array(test_dataset["list_classes"][:])

    return x_train, y_train, x_validation, y_validation, classes
