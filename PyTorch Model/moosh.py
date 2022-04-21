import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Définition d'une structure avec les épaisseurs comme arguments 
def StructureHeight(height, periods):
    Eps = np.tile([2, 3], (1, periods))
    Eps = np.insert(Eps, 0, 1)
    Eps = np.append(Eps, 1)
    Mu = np.ones(2 * periods)
    return Eps, Mu, height

# Définition d'une structure avec des épaisseurs aléatoires
def random_structure_imp(periods):
    Eps = np.tile([2, 3], (1, periods))
    Eps = np.insert(Eps, 0, 1)
    Eps = np.append(Eps, 1)
    Mu = np.ones(2 * periods + 2)
    height = []
    for i in range(2 * periods):
        randHeight = np.random.randint(40, 200)
        height.append(randHeight)
    height = np.asarray(height)
    Height = np.insert(height, 0, 100)
    Height = np.insert(Height, len(Height), 100)
    return Eps, Mu, Height

# Structure miroir de Bragg
def StructureBragg(periods):
    Eps = np.tile([2, 3], (1, periods))
    Eps = np.insert(Eps, 0, 1)
    Eps = np.append(Eps, 1)

    Mu = np.ones((2 * periods + 2))

    Height = np.tile([600 / (4 * np.sqrt(2)), 600 / (4 * np.sqrt(3))], (1, periods))
    Height = np.insert(Height, 0, 1600)
    Height = np.append(Height, 100)

    return Eps, Mu, Height

# Calcul des coefficients de réflexion
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

        re = np.ones((1000, 1), dtype=complex)

        for i in range(1000):
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

