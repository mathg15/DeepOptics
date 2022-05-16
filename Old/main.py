import numpy as np
import moosh as ms
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Train set

X = np.load('X_train_Bragg_Random.npy')
X = X.reshape(X.shape[0], -1)
y = np.load('y_train_Bragg_Random.npy')
y = y.reshape(y.shape[0], -1)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1)

X_train = X_train.reshape(9000, 100)
X_train = torch.from_numpy(X_train)

y_train = y_train.astype(int)
y_train = y_train.reshape(9000, 1)
y_train = torch.from_numpy(y_train)

# print(X_train.shape, y_train.shape)
# print(X_train[0])

X_val = X_val.reshape(1000, 100)
X_val = torch.from_numpy(X_val)

y_val = y_val.astype(int)
y_val = y_val.reshape(1000)
y_val = torch.from_numpy(y_val)

dataset_train = TensorDataset(X_train, y_train)  # create your datset
dataloader_train = DataLoader(dataset_train, batch_size=100)  # create your dataloader

dataset_val = TensorDataset(X_val, y_val)  # create your datset
dataloader_val = DataLoader(dataset_val, batch_size=1)  # create your dataloader

# Train set

X_test = np.load('X_test_Bragg_Random.npy')
X_test = X_test.reshape(X_test.shape[0], -1)
y_test = np.load('y_test_Bragg_Random.npy')
y_test = y_test.reshape(y_test.shape[0], -1)

X_test = torch.from_numpy(X_test)
y_test = torch.from_numpy(y_test)

dataset_test = TensorDataset(X_test, y_test)  # create your datset
dataloader_test = DataLoader(dataset_test, batch_size=1)  # create your dataloader


class Bragg_Class_Conv1D(nn.Module):
    def __init__(self):
        super(Bragg_Class_Conv1D, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(100, 1500),
            nn.LeakyReLU(),
            nn.Linear(1500, 1000),
            nn.LeakyReLU(),
            nn.Linear(1000, 200),
            nn.LeakyReLU(),
            nn.Linear(200, 50),
            nn.LeakyReLU(),
            nn.Linear(50, 1),
            nn.Sigmoid()

        )

    def forward(self, x):
        x = self.linear(x)
        return x


Model = Bragg_Class_Conv1D()
Model = Model.double()
# print(Model.eval())

N_epoch = 2
learning_rate = 8E-5

optimizer = torch.optim.AdamW(Model.parameters(), lr=learning_rate)
loss_function = nn.MSELoss()

for features_val, labels_val in dataloader_val:
    labels_val_pred = Model(features_val)

losses = []
val_losses = []
i = 0

for epoch in range(N_epoch):  # Loop over epochs
    running_loss = 0.0

    for features, labels in dataloader_train:

        features = features.double()
        labels = labels.double()

        # Forward Propagation
        labels_pred = Model(features)
        labels_val_pred = Model(features_val)

        # Loss computation
        loss = loss_function(labels_pred, labels)

        # Save loss for future analysis
        losses.append(loss.item())
        val_loss = loss_function(labels_val_pred, labels_val)
        val_losses.append(val_loss.item())

        # Erase previous gradients
        optimizer.zero_grad()

        # Compute gradients (backpropagation)
        loss.backward()

        # Weight update
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 10 == 9:
            print('[Epoque : %d, iteration: %5d] loss: %.4f' %
                  (epoch + 1, i + 1, running_loss / 10))
            running_loss = 0.0
        i += 1

print('Training done')


def display(losses, label='Training loss function'):
    # Display loss evolution
    fig, axes = plt.subplots(figsize=(8, 6))
    axes.plot(losses, 'r-', lw=2, label=label)
    axes.plot(val_losses, 'b-', lw=2, label='Validation loss function')
    axes.set_xlabel('N iterations', fontsize=18)
    axes.set_ylabel('Loss', fontsize=18)
    plt.legend(loc='upper right', fontsize=16)
    plt.show()


# # Display loss evolution
display(losses)


def accuracy(model, dataloader):
    correct = 0
    total = 0
    # No need to compute gradients here
    with torch.no_grad():
        for features, labels in dataloader:
            # Forward propagation to get predictions
            features = features
            labels = labels.long()
            outputs = model(features)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100.0 * correct / total


print(f'Accuracy : {accuracy(Model, dataloader_test):.2f}')
#
predictions = Model(X_test)
print(f'predictions : {predictions}')
print(f'true : {y_test}')

