import torch.nn as nn
import moosh as ms
import numpy as np
import torch
import matplotlib.pyplot as plt

# Dataset

def DataSetGen():
    # np.random.seed(0)
    structure = ms.random_structure_imp(5)
    _, _, height = structure
    coef = ms.Reflection(structure)
    re = coef.genSpectrum()
    Height = np.delete(height, 0)
    Height = np.delete(Height, len(Height) - 1)
    return re, Height.astype(float)


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


X_train, y_train = TrainingDataLoad(100)
X_test, y_test = TrainingDataLoad(1)

print(y_test)
dataset = []
for i in range(len(X_train)):
    spec = X_train
    H = y_train
    dataset.append((spec, H))

test_set = []
for i in range(len(X_test)):
    spec = X_test
    H = y_test
    test_set.append((spec, H))


class Custom_Dataset(torch.utils.data.dataset.Dataset):
    def __init__(self, _dataset):
        self.dataset = _dataset

    def __getitem__(self, index):
        example, target = self.dataset[index]
        return np.array(example), target

    def __len__(self):
        return len(self.dataset)


train_loader = torch.utils.data.DataLoader(dataset=Custom_Dataset(dataset),
                                           batch_size=5,
                                           shuffle=False)

test_loader = torch.utils.data.DataLoader(dataset=Custom_Dataset(test_set),
                                          batch_size=100,
                                          shuffle=False)
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(100, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 10),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        x = self.linear(x)
        return x


model = Model()
model = model.double()
print(model.eval())
N_epochs = 100
learning_rate = 1E-4
loss_function = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)



def train(model, data_loader, opt, n_epochs):
    losses = []
    i = 0
    for epoch in range(n_epochs):  # Loop over epochs
        running_loss = 0.0

        for features, labels in data_loader:

            Out_pred = model(features)
            loss = loss_function(Out_pred, labels)
            losses.append(loss.item())
            # Erase previous gradients
            opt.zero_grad()
            # Compute gradients (backpropagation)
            loss.backward()
            # Weight update
            opt.step()

            # print statistics
            running_loss += loss.item()
            if i % 10 == 9:
                print('[Epoch : %d, iteration: %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 10))
                running_loss = 0.0
            i += 1

    print('Training done')
    return losses


losses = train(model, train_loader, optimizer, N_epochs)


def display(losses, label='Training loss function'):
    # Display loss evolution
    fig, axes = plt.subplots(figsize=(8, 6))
    axes.plot(losses, 'r-', lw=2, label=label)
    axes.set_xlabel('N iterations', fontsize=18)
    axes.set_ylabel('Loss', fontsize=18)
    plt.legend(loc='upper right', fontsize=16)
    plt.show()


# # Display loss evolution
display(losses)

#### Bragg miror ####
structure = ms.StructureBragg(5)
_,_, Height_Bragg = structure
coef1 = ms.Reflection(structure)
R = coef1.genSpectrum()
R = np.transpose(R)
R = torch.from_numpy(R)


pred = model.forward(R)
pred = pred.detach().numpy()


Pred = np.insert(pred[0], 0, 1600)
Pred = np.insert(Pred, 11, 100)

print(f'Pred :{Pred}')
print(f'True : {Height_Bragg}')

st = ms.StructureHeight(Pred, 5)
coef1 = ms.Reflection(st)
spec_pred = coef1.genSpectrum()

rangeLambda = np.linspace(400, 800, 100)
plt.plot(rangeLambda, R.T, label='True')
plt.plot(rangeLambda, spec_pred, label='Predict')
plt.ylabel("Reflection")
plt.xlabel("Wavelength")
plt.legend(loc='best')
plt.show()
