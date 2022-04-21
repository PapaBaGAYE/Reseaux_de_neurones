import numpy as np
from projet_deepl.tenseurs.tenseur import Tenseur
from projet_deepl.layers.activation import Activation


def tanh(x: Tenseur) -> Tenseur:
    """
    Determine la relu de x

    Args:
        x (Tenseur)

    Returns:
        La tanh(x) : Tenseur
    """
    return np.tanh(x)


def tanh_prime(x: Tenseur) -> Tenseur:
    """Dérivée de la fonction tangente_hyperbolique

    Args:
        x (Tenseur): Sorties de couche linéaire stockées

    Returns:
        Tenseur: Les informations du backward
    """
    y = tanh(x)
    return 1 - y**2


class TanActivation(Activation):
    def __init__(self, tanh, tanh_prime):
        super().__init__(tanh, tanh_prime)
