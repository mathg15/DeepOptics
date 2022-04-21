import torch.nn as nn
import moosh as ms
import numpy as np
import torch
import matplotlib.pyplot as plt
import time

# GPU device
assert torch.cuda.is_available()
cuda_device = torch.device("cuda:0")
cpu_device = torch.device("cpu")

# Dataset
X_train = np.load('X_train_new_h.npy')
y_train = np.load('y_train_new_h.npy')

dataset = []
for i in range(len(X_train)):
    spec = X_train
    H = y_train
    dataset.append((spec, H))


class Custom_Dataset(torch.utils.data.dataset.Dataset):
    def __init__(self, _dataset):
        self.dataset = _dataset

    def __getitem__(self, index):
        example, target = self.dataset[index]
        return np.array(example), target

    def __len__(self):
        return len(self.dataset)


train_loader = torch.utils.data.DataLoader(dataset=Custom_Dataset(dataset),
                                           batch_size=20,
                                           shuffle=True)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = nn.Sequential(

            nn.Linear(100, 256),
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


model = Model().to(cuda_device)
model = model.double()
print(model.eval())
N_epochs = 5
learning_rate = 1E-4
loss_function = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


def train(model, data_loader, opt, n_epochs):
    time0 = time.time()
    losses = []
    i = 0
    for epoch in range(n_epochs):
        running_loss = 0.0

        for features, labels in data_loader:
            features = features.to(cuda_device)
            labels = labels.to(cuda_device)
            Out_pred = model(features)
            loss = loss_function(Out_pred, labels)
            losses.append(loss.item())
            opt.zero_grad()
            loss.backward()
            opt.step()
            running_loss += loss.item()
            if i % 10 == 9:
                print(f'[Epoch : {epoch + 1}, iteration: {i + 1}]'
                      f' loss: {running_loss / 10 :.3f} '

                      )
                running_loss = 0.0
            i += 1
    time1 = time.time()
    print(f'Training done, time : {(time1 - time0) / 60:.2f} mins')
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
_, _, Height_Bragg = structure
coef1 = ms.Reflection(structure)
R = coef1.genSpectrum()
R = np.transpose(R)
R = torch.from_numpy(R)
R = R.to(cpu_device)

model = model.to(cpu_device)
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
