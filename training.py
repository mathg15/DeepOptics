import numpy as np
import moosh as ms
import neural_networks as nn
import modelObject as mod
import matplotlib.pyplot as plt


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

model = mod.Model()
model.add(nn.Net(100, 128))
model.add(nn.ReLU())
model.add(nn.Net(128, 64))
model.add(nn.ReLU())
model.add(nn.Net(64, 32))
model.add(nn.ReLU())
model.add(nn.Net(32, 16))
model.add(nn.ReLU())
model.add(nn.Net(16, 4))
model.add(nn.ReLU())

model.set(nn.MSELoss(),
          nn.Optimizer_SGD(learning_rate=1e-14, decay=0.0001),
          nn.Accuracy_Regression())

model.finalize()
model.train(X_train, y_train, epoch=1000, print_every=1, batch_size=80)

structure = ms.random_structure_imp()
_, _, h = structure
coef = ms.Reflection(structure)
re = coef.genSpectrum()

print(f'Predict: {model.predict(re.T)[0]}')
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

model.display_loss()

