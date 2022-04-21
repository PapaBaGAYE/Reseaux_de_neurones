from typing import Callable, Dict
from projet_deepl.tenseurs.tenseur import Tenseur
from projet_deepl.layers.layer import Layer

F = Callable[[Tenseur], Tenseur]


class Activation(Layer):
    def __init__(self, f: F, f_prime: F) -> None:
        self.params: Dict[str, Tenseur] = {}
        self.grads: Dict[str, Tenseur] = {}
        self.f = f
        self.f_prime = f_prime

    def forward(self, inputs: Tenseur) -> float:
        """
        Fonction forward

        Args:
            inputs  (Tenseur)

        Returns:
            la fonction (float)
        """
        self.inputs = inputs
        return self.f(self.inputs)

    def backward(self, grad: Tenseur) -> Tenseur:
        """
        Fonction backward

        Args:
            inputs  (Tenseur)

        Returns:
            la dérivée (Tenseur)
        """
        return self.f_prime(self.inputs) * grad
