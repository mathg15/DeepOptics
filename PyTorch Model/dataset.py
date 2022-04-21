import moosh as ms
import numpy as np
import time
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

t0 = time.time()
X_train, y_train = TrainingDataLoad(10000)

np.save('X_train_new_h.npy', X_train)
np.save('y_train_new_h.npy', y_train)
t1 = time.time()

print(f'Temps : {t1-t0}')
