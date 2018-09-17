# libs
import numpy

# project libs
import functions.loss
import functions.activation

import time


class LogisticRegression():
    

    def __init__(self, input_shape, loss_function='log', activation_function='sigmoid', d_part=0):
        
        # code from towarddatascience
        self.input_shape = input_shape

        # initializing w
        self.weights = numpy.zeros(self.input_shape)
        # init b
        self.bias = 0

        self.loss = loss_function
        self.activation = activation_function
        self.d_part = d_part
    
    def predict(self, input_features):
        """do a forward propagation (ŷ = w*x + b)
        
        Arguments:
            input_features {nmpy.array} -- array of inputs (must be flat)
        
        Returns:
            float -- probability between [0; 1]
        """

        # ŷ = w * x + b
        # output = weights * input + bias
        output = numpy.dot(self.weights, input_features)
        output = output + self.bias

        y_predict = self._activate(output)
        return y_predict

    def train(self, x_train_flat, y_train, x_validation_flat, y_validation, epoch, learning_rate, recognition_threshold):
        
        # for each epoch, we will first update the model on each images
        #     then, we will evaluate the model on the validation data
        train_acc = []
        vall_acc = []
        train_cost = []
        for epoch_nb in range(epoch):
            ##
            # training the network for each images
            ##
            train_correct = 0
            total_train = x_train_flat.shape[1]

            for position in range(total_train):
                # predict for this sample
                y_predict = self.predict(x_train_flat[:, position])

                """# save some data for training visualisation
                print('y_pred', y_predict)
                print('y_train', y_train[0, position])
                print('position', position)
                print('train_correct', train_correct)
                print("\n\n\n")
                time.sleep(5)"""

                threshold_output = y_predict                
                if(y_predict <= recognition_threshold): # <= 0.3 --> 0
                    threshold_output = 0
                if(y_predict >= 1 - recognition_threshold): # >= 0.7 --> 1
                    threshold_output = 1

                if(threshold_output == y_train[0, position]):
                    train_correct += 1
                
                # calculate loss
                loss = self._calculate_loss(y_train[0, position], y_predict)

                # since there is only one sample, our cost = loss
                cost = loss
                train_cost.append(cost)

                # and now we update weights and bias
                self._update_weights(cost, learning_rate, y_predict, y_train[0, position], x_train_flat[:, position])
            
            ###
            # evaluating validation set
            ###
            validation_correct = 0
            total_validation = x_validation_flat.shape[1]

            for pos_validation in range(total_validation):
                #predict for this sample
                y_predict = self.predict(x_train_flat[:, pos_validation])

                # save some data for training visualisation
                threshold_output = y_predict                
                if(y_predict <= recognition_threshold): # <= 0.3 --> 0
                    threshold_output = 0
                if(y_predict >= 1 - recognition_threshold): # >= 0.7 --> 1
                    threshold_output = 1

                if(threshold_output == y_validation[0, pos_validation]):
                    validation_correct += 1                

                
            
            train_acc.append(train_correct / total_train)
            vall_acc.append(validation_correct / total_validation)
        
        history = (train_acc, vall_acc, train_cost)
        return history


    def _update_weights(self, cost, learning_rate, y_predict, y_truth, input_data):
        
        # dz = a - y | a = sigma | sigma = sigmoid(output) | sigmoid(output) = y_predict -> a = y_predict
        gradient = y_predict - y_truth

        gradient_weights = input_data * gradient

        self.weights = self.weights - (learning_rate * gradient_weights)

        self.bias = self.bias - (learning_rate * gradient)

    def _calculate_loss(self, y_truth, y_predict):
        """return the loss between y and ŷ
        
        Arguments:
            y_truth {float} -- y
            y_predict {float} -- ŷ
        
        Returns:
            float -- loss
        """
        if(self.loss == 'ms'):
            loss = functions.loss.mean_squared(y_truth, y_predict)
        
        if(self.loss == 'abs'):
            loss = functions.loss.absolute(y_truth, y_predict)
        
        if(self.loss == "huber"):
            loss = functions.loss.huber(y_truth, y_predict, self.d_part)

        if(self.loss == 'log'):
            loss = functions.loss.cross_enthropy(y_truth, y_predict)
        
        return loss
    
    #TODO other activation
    def _activate(self, output):
        if(self.activation == 'sigmoid'):
            result = functions.activation.sigmoid(output)
        return result

