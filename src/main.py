# python default package
import os

# libs
import numpy
import matplotlib.pyplot as plt

# project packages
import filehandler
import preprocessing
import model.log_reg as log_reg

# Loading the dataset
path = "../"
x_train, y_train, x_validation, y_validation, classes = filehandler.import_dataset(
    path)


# Example of a picture
"""index = 20
plt.imshow(x_train[index])
plt.show()
print("y = " + str(y_train[:, index]) + ", it's a '" + classes[numpy.squeeze(y_train[:, index])].decode("utf-8") + "' picture.")
"""

# preprocessing the dataset
train_flat, validation_flat = preprocessing.preprocess_images(x_train, x_validation)


# creating our model
input_shape = train_flat.shape[0]
log_regression = log_reg.LogisticRegression(input_shape)

# training our model
epochs = 500
learning_rate = 0.0001 
# if the probability is below 0.3 or above 0.7, the sample is "correctly predicted"
recognition_threshold = 0.3

history = log_regression.train(train_flat, y_train, validation_flat, y_validation, epochs, learning_rate, recognition_threshold)


# and now let's see the result

print("last cost:", history[2][-1])

plt.plot(history[2])
plt.title('Cost evolution')
plt.ylabel('Cost')
plt.xlabel('Iterations')
plt.show()

print("last validation:", history[1][-1])
print("last train:", history[0][-1])
plt.plot(history[1], label='validation accuracy')
plt.plot(history[0], label='train accuracy')
plt.title('accuracy evolution')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend()
plt.show()