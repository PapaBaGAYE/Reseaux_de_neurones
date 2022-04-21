import numpy as np
from projet_deepl.tenseurs.tenseur import Tenseur
from projet_deepl.layers.activation import Activation


def sig(x: Tenseur) -> Tenseur:
    """Fonction d'activation sigmoid

    Args:
        x (Tenseur) : Sortie de la couche linéaire

    Returns:
        Tenseur: Les informations du forward
    """
    return 1.0 / (1 + np.exp(-x))


def sig_prime(x: Tenseur) -> Tenseur:
    """Dérivée de la fonction sigmoid

    Args:
        x (Tenseur) : Sorties de couche linéaire stockées

    Returns:
        Tenseur: Les informations du backward
    """
    return sig(x) * (1 - sig(x))


class SigActivation(Activation):
    def __init__(self, sig, sig_prime):
        super().__init__(sig, sig_prime)
