import numpy


def preprocess_images(x_train, x_validation):
    """preprocess pictures
    
    Arguments:
        x_train {numpy.array} -- array of train features
        x_validation {numpy.array} -- array of test features
    
    Returns:
        train_flat, validation_flat -- features normalized and flattened
    """
    
    # Flatten the pictures
    train_flat = x_train.reshape(x_train.shape[0], -1).T
    validation_flat = x_validation.reshape(x_validation.shape[0], -1).T


    # normalising
    train_flat = train_flat.astype(numpy.float32)
    train_flat /= 255.

    validation_flat = validation_flat.astype(numpy.float32)
    validation_flat /= 255.
    
    return train_flat, validation_flat
