import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import torch.nn as nn

X = np.load('X_randH.npy')
y = np.load('y_randH.npy')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)
# print(f'For features => Train size : {X_train.shape}, Test size : {X_test.shape}, Val size : {X_val.shape}')
# print(f'For labels => Train size : {y_train.shape}, Test size : {y_test.shape}, Val size : {y_val.shape}')

# Train set
batch_train = 20
X_train = torch.from_numpy(X_train)
# print(X_train.shape)
y_train = torch.from_numpy(y_train)
dataset_train = TensorDataset(X_train, y_train)  # create your datset
dataloader_train = DataLoader(dataset_train, batch_size=batch_train)  # create your dataloader

# Test set
batch_test = 50
X_test = torch.from_numpy(X_test)
y_test = torch.from_numpy(y_test)
dataset_test = TensorDataset(X_test, y_test)  # create your datset
dataloader_test = DataLoader(dataset_test, batch_size=batch_test)

# Validation set
batch_val = 1
X_val = torch.from_numpy(X_val)
y_val = torch.from_numpy(y_val)
dataset_val = TensorDataset(X_val, y_val)  # create your datset
dataloader_val = DataLoader(dataset_val, batch_size=batch_val)  # create your dataloader

X_false = torch.rand((34000, 50))
y_false = torch.zeros((34000, 1))
dataset_false = TensorDataset(X_false, y_false)  # create your datset
dataloader_false = DataLoader(dataset_val, batch_size=100)

dataset_temp = TensorDataset(X_train, X_false)
dataloader_temp = DataLoader(dataset_temp, batch_size=100)


# print(X_false.shape)


class DiscriminatorNet(torch.nn.Module):
    """
    A three hidden-layer discriminative neural network
    """

    def __init__(self):
        super(DiscriminatorNet, self).__init__()
        n_features = 50
        n_out = 1

        self.hidden0 = nn.Sequential(
            nn.Linear(n_features, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        self.hidden1 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        self.out = nn.Sequential(
            torch.nn.Linear(256, n_out),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        return x


class GeneratorNet(torch.nn.Module):


    def __init__(self):
        super(GeneratorNet, self).__init__()
        n_features = 100
        n_out = 50

        self.hidden0 = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.LeakyReLU(0.2)
        )
        self.hidden1 = nn.Sequential(
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2)
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2)
        )

        self.out = nn.Sequential(
            nn.Linear(1024, n_out),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        return x


generator = GeneratorNet()
generator.double()
discriminator = DiscriminatorNet()
discriminator.double()

optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=1E-4, weight_decay=0.9)
optimizer_g = torch.optim.Adam(discriminator.parameters(), lr=1E-4, weight_decay=0.9)
loss_function = nn.BCELoss()


def noise(size):
    n = torch.randn(size, 100)
    return n


def train_generator(false_data):
    # Reset gradients
    optimizer_g.zero_grad()
    # Noise into fake data
    false_data = false_data.double()
    fake_pred = discriminator(false_data)
    # Loss calculation
    loss = loss_function(fake_pred, torch.ones((batch_train, 1), dtype=float))
    loss.backward()
    optimizer_g.step()
    return loss.item()


def train_discrim(real_features, false_features):
    optimizer_d.zero_grad()

    real_features = real_features.double()
    # Real preds
    real_preds = discriminator(real_features)
    # Real labels
    real_labels = torch.ones((batch_train, 1), dtype=float)
    # Loss
    real_loss = loss_function(real_preds, real_labels)
    # print(f'Real loss {real_loss.item():.2f}')
    real_loss.backward()

    # print(features.shape)
    false_features = false_features.double()
    # Real preds
    false_preds = discriminator(false_features)
    # Real labels
    false_labels = torch.ones((batch_train, 1), dtype=float)
    # Loss
    false_loss = loss_function(false_preds, false_labels)
    # print(f'Fake loss {false_loss.item():.2f}')
    false_loss.backward()

    optimizer_d.step()

    loss_sum = real_loss.item() + false_loss.item()
    return loss_sum


def display(losses, label='Training loss function'):
    # Display loss evolution
    fig, axes = plt.subplots(figsize=(8, 6))
    axes.plot(losses, 'r-', lw=2, label=label)
    axes.set_xlabel('N iterations', fontsize=18)
    axes.set_ylabel('Loss', fontsize=18)
    plt.legend(loc='upper right', fontsize=16)
    plt.show()


n_epochs = 10
losses_d = []
losses_g = []


for epoch in range(n_epochs):
    for features, _ in dataloader_train:
        # 1_ Train discriminator

        real_features = features.double()
        # Generate fake data
        N = noise(batch_train).double()
        fake_data = generator(N).detach()
        # Train D
        d_loss = train_discrim(real_features, fake_data)
        # print(f'Discri loss : {d_loss}')
        losses_d.append(d_loss)

        # 2_ Train generator
        # Generate fake data
        fake_data = generator(noise(batch_train).double())
        # Train G
        g_loss = train_generator(fake_data)
        # print(f'Generator loss : {g_loss}')
        losses_g.append(g_loss)

display(losses_g, 'Generator')
display(losses_d, 'Discriminator')

test_noise = noise(1).double()
a = generator(test_noise)
a = a.detach().numpy()
x = np.linspace(400, 800, 50)
plt.plot(x, a.T)
plt.show()
# print(a)
