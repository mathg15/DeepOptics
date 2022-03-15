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


class Activation_Linear:
    # Forward pass
    def forward(self, inputs):
        # Just remember values
        self.inputs = inputs
        self.outputs = inputs

    # Backward pass
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()

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


class Layer_Dropout:
    # Init
    def __init__(self, rate):
        # Store rate, we invert it as for example for dropout
        # of 0.1 we need success rate of 0.9
        self.rate = 1 - rate

    # Forward pass
    def forward(self, inputs, training):
        # Save input values
        self.inputs = inputs
        # If not in the training mode - return values
        if not training:
            self.output = inputs.copy()
            return
        # Generate and save scaled mask
        self.binary_mask = np.random.binomial(1, self.rate,
                                              size=inputs.shape) / self.rate
        # Apply mask to output values
        self.output = inputs * self.binary_mask

    # Backward pass
    def backward(self, dvalues):
        # Gradient on values
        self.dinputs = dvalues * self.binary_mask


# Loss
class Loss:

    def remember_trainable_layers(self, trainable_layers):
        self.trainable_layers = trainable_layers

    def calculate(self, output, y):
        samples_losses = self.forward(output, y)
        data_loss = np.mean(samples_losses)

        self.accumulated_sum += np.sum(samples_losses)
        self.accumulated_count += len(samples_losses)
        return data_loss

    def calculate_accumulated(self):
        data_loss = self.accumulated_sum / self.accumulated_count
        return data_loss

    def new_pass(self):
        self.accumulated_sum = 0.
        self.accumulated_count = 0.


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
    def __init__(self, learning_rate=1., decay=0.):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0

    # Call once before any parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                                         (1. / (1. + self.decay * self.iterations))

    # Update parameters
    def update_params(self, layer):
        layer.weights += -self.current_learning_rate * layer.dweights
        layer.biases += -self.current_learning_rate * layer.dbiases
        # Call once after any parameter updates

    def post_update_params(self):
        self.iterations += 1


class Accuracy:
    def calculate(self, predictions, y):
        comparisons = self.compare(predictions, y)
        accuracy = np.mean(comparisons)

        self.accumulated_sum += np.sum(comparisons)
        self.accumulated_count += len(comparisons)
        return accuracy

    def calculate_accumulated(self):
        accuracy = self.accumulated_sum / self.accumulated_count
        return accuracy

    def new_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0


class Accuracy_Regression(Accuracy):
    def __init__(self):
        self.precision = None

    def init(self, y, reinit=False):
        if self.precision is None or reinit:
            self.precision = np.std(y) / 250

    def compare(self, predictions, y):
        return np.absolute(predictions - y) < self.precision
