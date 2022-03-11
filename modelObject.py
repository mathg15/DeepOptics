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

            for layer in self.trainable_layers:
                self.optimizer.update_params(layer)

            if not step % print_every or step == train_steps - 1:
                print(f'Step : {step}, ' +
                      f'Acc : {accuracy:.3f}, ' +
                      f'Loss : {loss:.3f}')

        epoch_loss = self.loss.calculate_accumulated()
        print(f'training' +
                  f'Loss : {epoch_loss}')

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


# Dataset

def DataSetGen():
    structure = ms.random_structure_imp()
    _, _, height = structure
    coef = ms.Reflection(structure)
    re = coef.genSpectrum()
    return re, height


def TrainingDataLoad(train_size):
    X_train = []
    Y_train = []
    for i in range(train_size):
        x_train, y_train = DataSetGen()
        X_train.append(x_train)
        Y_train.append(y_train)

    X_train = np.asarray(X_train)
    X_train = X_train.reshape(X_train.shape[0], -1)

    Y_train = np.asarray(Y_train)
    Y_train = Y_train.reshape(Y_train.shape[0], -1)

    return X_train, Y_train


X_train, y_train = TrainingDataLoad(1000)


model = Model()
model.add(nn.Net(100, 10))
model.add(nn.ReLU())
# model.add(nn.Net(6, 4))
# model.add(nn.ReLU())

model.set(nn.MSELoss(),
          nn.Optimizer_SGD(learning_rate=0.1),
          Accuracy_Regression())

model.finalize()
model.train(X_train, y_train, epoch=1000, print_every=100, batch_size=20)

structure = ms.random_structure_imp()
_,_,h = structure
coef = ms.Reflection(structure)
re = coef.genSpectrum()

print(f'Predict: {model.predict(re.T)}')
print(f' True : {h}')

st = ms.StructureHeight(model.predict(re.T)[0])

coef1 = ms.Reflection(st)
pred = coef1.genSpectrum()

rangeLambda = np.linspace(400, 800, 100)
plt.plot(rangeLambda, re, label='True')
plt.plot(rangeLambda, pred, label='Predict')
plt.ylabel("Reflection")
plt.xlabel("Wavelength")
plt.legend(loc='best')
plt.show()
