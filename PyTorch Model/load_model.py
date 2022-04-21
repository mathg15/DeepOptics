import torch
import torch.nn as nn
import moosh as ms
import numpy as np
import matplotlib.pyplot as plt
cpu_device = torch.device("cpu")


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


model = Model()
model = model.double()
model.load_state_dict(torch.load('model_weights.pth'))

print(model.eval())

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
