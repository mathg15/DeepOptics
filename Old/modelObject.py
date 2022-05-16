import neural_networks as nn
import numpy as np
import moosh as ms
import matplotlib.pyplot as plt


class Layer_Input:
    def forward(self, inputs):
        self.outputs = inputs


class Model:
    def __init__(self):
        # Create a list of network objects
        self.layers = []

    # Add objects to the model
    def add(self, layer):
        self.layers.append(layer)

    def set(self, loss_function, optimizer, accuracy):
        self.loss = loss_function
        self.optimizer = optimizer
        self.accuracy = accuracy

    def finalize(self):
        self.input_layer = Layer_Input()
        layer_count = len(self.layers)
        self.trainable_layers = []

        for i in range(layer_count):

            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i + 1]

            elif i < layer_count - 1:
                self.layers[i].prev = self.layers[i - 1]
                self.layers[i].next = self.layers[i + 1]

            else:
                self.layers[i].prev = self.layers[i - 1]
                self.layers[i].next = self.loss
                self.output_layer_activation = self.layers[i]

            if hasattr(self.layers[i], "weights"):
                self.trainable_layers.append(self.layers[i])

    def train(self, X, y, epoch, print_every, batch_size=None):
        self.save_loss = []
        self.epoch = epoch
        self.accuracy.init(y=y)
        train_steps = 1

        if batch_size is not None:
            train_steps = len(X) // batch_size
            if train_steps * batch_size < len(X):
                train_steps += 1

        for epoch in range(1, epoch + 1):

            print(f'Epoch : {epoch}')
            self.loss.new_pass()
            self.accuracy.new_pass()

            for step in range(train_steps):
                if batch_size is None:
                    batch_X = X
                    batch_y = y

                else:
                    batch_X = X[step * batch_size:(step + 1) * batch_size]
                    batch_y = y[step * batch_size:(step + 1) * batch_size]

                output = self.forward(batch_X)

                data_loss = self.loss.calculate(output, batch_y)
                loss = data_loss
                predictions = self.output_layer_activation.predictions(output)
                accuracy = self.accuracy.calculate(predictions, batch_y)

                self.backward(output, batch_y)

            self.optimizer.pre_update_params()
            for layer in self.trainable_layers:
                self.optimizer.update_params(layer)
            self.optimizer.post_update_params()

            if not step % print_every or step == train_steps - 1:
                print(f'Step : {step}, ' +
                      f'Acc : {accuracy:.3f}, ' +
                      f'Loss : {loss:.3f}' +
                      f'lr: {self.optimizer.current_learning_rate}')

            self.save_loss.append(loss)
            epoch_loss = self.loss.calculate_accumulated()
            print(f'training' +
                  f'Loss : {epoch_loss:.3f}')

            self.save_loss.append(loss)

    def forward(self, X):
        self.input_layer.forward(X)

        for layer in self.layers:
            layer.forward(layer.prev.outputs)

        return layer.outputs

    def backward(self, output, y):
        self.loss.backward(output, y)

        for layer in reversed(self.layers):
            layer.backward(layer.next.dinputs)

    def predict(self, X):
        output = self.forward(X)
        return output

    def display_loss(self):
        nb_loss_point = len(self.save_loss)
        epoch = np.linspace(0, self.epoch, nb_loss_point)
        Loss = np.asarray(self.save_loss)
        plt.plot(epoch, Loss)
        plt.show()
