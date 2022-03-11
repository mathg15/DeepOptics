import numpy as np
from numpy.random import rand


# Activation function
class ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = np.maximum(0, inputs)  # ReLU function : f(x) = max(0,x)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0  # Derivative of the ReLU function

    def predictions(self, outputs):
        return outputs


# Neural network

class Net:
    def __init__(self, input_Layer, output_Layer):
        self.weights = rand(input_Layer, output_Layer)  # Random weights for the fist layer
        self.biases = np.zeros((1, output_Layer))  # Set biases to 0

    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        # Gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)


# Loss
class Loss:

    def remember_trainable_layers(self, trainable_layers):
        self.trainable_layers = trainable_layers

    def calculate(self, output, y):
        samples_losses = self.forward(output, y)
        data_loss = np.mean(samples_losses)
        return data_loss

        self.accumulated_sum += np.sum(samples_losses)
        self.accumulated_count += len(samples_losses)
        return data_loss

    def calculate_accumulated(self):
        data_loss = self.accumulated_sum / self.accumulated_count
        return data_loss

    def new_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0.1


# Loss function
class MSELoss(Loss):
    def forward(self, y_pred, y_true):
        sample_losses = np.mean((y_true - y_pred) ** 2, axis=-1)
        return sample_losses

    def backward(self, dvalues, true):
        samples = len(dvalues)
        outputs = len(dvalues[0])
        self.dinputs = -2 * (true - dvalues) / outputs
        return self.dinputs / samples


class Optimizer_SGD:
    # Initialize optimizer - set settings,
    # learning rate of 1. is default for this optimizer
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    # Update parameters
    def update_params(self, layer):
        layer.weights += -self.learning_rate * layer.dweights
        layer.biases += -self.learning_rate * layer.dbiases
