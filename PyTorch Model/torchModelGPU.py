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

# Création du dataset
X_train = np.load('X_train_new_h.npy') # Features (spectre)
y_train = np.load('y_train_new_h.npy') # Labels (épaisseur de couche)

dataset = [] # Liste permettant de créer des paires (spectres, épaisseurs)
for i in range(len(X_train)):
    spec = X_train
    H = y_train
    dataset.append((spec, H))

# On transforme la liste en un tensor
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

# On génère un réseau de neurones
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = nn.Sequential(

            nn.Linear(100, 256), # Input
            nn.LeakyReLU(), # Fonction d'activation
            nn.Linear(256, 128), # Hidden Layer
            nn.LeakyReLU(), # Fonction d'activation
            nn.Linear(128, 64), # Hidden Layer
            nn.LeakyReLU(), # Fonction d'activation
            nn.Linear(64, 32), # Hidden Layer
            nn.LeakyReLU(), # Fonction d'activation
            nn.Linear(32, 16), # Hidden Layer
            nn.LeakyReLU(), # Fonction d'activation
            nn.Linear(16, 10), # Output
            nn.LeakyReLU(), # Fonction d'activation
        )

    def forward(self, x): # Propagation vers l'avant
        x = self.linear(x)
        return x


model = Model().to(cuda_device) # Changer par cpu_device pour calculer sur CPU 
model = model.double() # type Float64 
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
structure = ms.StructureBragg(5) # Génération de la structure du miroir de Bragg d'indice (2, 3)
_, _, Height_Bragg = structure
coef1 = ms.Reflection(structure)
R = coef1.genSpectrum() # Spectre de réflexion du miroir de Bragg en fonction de lambda ( 400nm à 800nm )
R = np.transpose(R)
R = torch.from_numpy(R) # Array to Tensor
R = R.to(cpu_device) # Faut pas changer le device

model = model.to(cpu_device) # Ici non plus
pred = model.forward(R) # On fait passer le spectre à travers le N.N.
pred = pred.detach().numpy() # Tensor to array

# On rajoute les couches d'air pour tracer le spectre
Pred = np.insert(pred[0], 0, 1600)
Pred = np.insert(Pred, 11, 100)

print(f'Pred :{Pred}')
print(f'True : {Height_Bragg}')

# Moosh pour tracer le spectre
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
