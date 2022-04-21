import numpy as np
from projet_deepl.tenseurs.tenseur import Tenseur
from projet_deepl.layers.activation import Activation


def relu(x: Tenseur) -> Tenseur:
    """
    Determine la relu de x

    Args:
        x (Tenseur)

    Returns:
        La relu(x) : Tenseur
    """


def relu_prime(x):
    """Couche d'activation

    Args:
        x (Tenseur)

    Returns:
        x (Tenseur)
    """
    x[x <= 0] = 0
    x[x > 0] = 1
    return x


class ReluActivation(Activation):
    def __init__(self, relu, relu_prime):
        super().__init__(relu, relu_prime)
