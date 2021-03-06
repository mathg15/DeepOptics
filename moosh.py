import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def Layer(Epsilon, Mu, Height):
    """
    :param Epsilon: Permittivity of the layer
    :param Mu: Permeability of the layer
    :param Height: Height of the layer
    :return: Array like [Epsilon, Mu, Height]
    """
    return np.array([[Epsilon, Mu, Height]])


def Structure(*args):
    """
    Create a multi-layers structure

    :param args: layers of the structure
    :return: array for each parameter of the layers

    Example :

    Layer1 = [1,1,100]; Layer2 = [2,1,100]

    Return : Eps = [1, 2]; Mu = [1, 1}; Height = [100, 100]
    """
    structure = args[0]
    for i in range(len(args) - 1):
        structure = np.append(structure, args[i + 1], axis=0)
    epsStruc = structure[:, 0]
    muStruc = structure[:, 1]
    heightStruc = structure[:, 2]
    return epsStruc, muStruc, heightStruc


def StructureHeight(height, periods):
    Eps = np.tile([2, 3], (1, periods))
    Eps = np.insert(Eps, 0, 1)
    Eps = np.append(Eps, 1)
    Mu = np.ones(2 * periods + 2)
    return Eps, Mu, height


def random_structure_imp(periods):
    Eps = np.tile([2, 3], (1, periods))
    Eps = np.insert(Eps, 0, 1)
    Eps = np.append(Eps, 1)
    Mu = np.ones(2 * periods + 2)
    height = []
    for i in range(2 * periods):
        randHeight = np.random.randint(1, 350)
        height.append(randHeight)
    height = np.asarray(height)
    Height = np.insert(height, 0, 100)
    Height = np.insert(Height, len(Height), 100)
    return Eps, Mu, Height


def StructureBragg(periods):
    Eps = np.tile([2, 3], (1, periods))
    Eps = np.insert(Eps, 0, 1)
    Eps = np.append(Eps, 1)

    Mu = np.ones((2 * periods + 2))

    Height = np.tile([600 / (4 * np.sqrt(2)), 600 / (4 * np.sqrt(3))], (1, periods))
    Height = np.insert(Height, 0, 1600)
    Height = np.append(Height, 100)

    return Eps, Mu, Height


class Reflection:

    def __init__(self, structure):
        self.structure = structure

    def coefficient(self, _lambda):
        Epsilon, Mu, hauteur = self.structure
        n = np.sqrt(Epsilon)

        tmp = np.tan(2 * np.pi * n * hauteur / _lambda)
        Z = n[-1]
        for k in range(np.size(Epsilon) - 1, 0, -1):
            Z = (Z - 1j * n[k] * tmp[k]) / (1 - 1j * tmp[k] * Z / n[k])
        r = (n[0] - Z) / (n[0] + Z)
        return r

    def spectrum(self):
        """
        :param angle: incident angle
        :return: Reflection vs wave length (400nm to 800nm)
        """
        rangeLambda = np.linspace(400, 800, 100)

        re = np.ones((100, 1), dtype=complex)

        for i in range(100):
            lambda_ = rangeLambda[i]
            re[i] = self.coefficient(lambda_)

        plt.title(f"Reflection spectrum ")
        plt.plot(rangeLambda, np.abs(re))
        plt.ylabel("Reflection")
        plt.xlabel("Wavelength")
        plt.grid(True)
        plt.show()

    def genSpectrum(self):
        rangeLambda = np.linspace(400, 800, 100)

        re = np.ones((100, 1), dtype=complex)

        for i in range(100):
            lambda_ = rangeLambda[i]
            re[i] = self.coefficient(lambda_)
        re = np.abs(re) ** 2
        return re
