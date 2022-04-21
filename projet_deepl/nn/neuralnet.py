from typing import Sequence, Iterator, Tuple

from projet_deepl.layers.layer import Layer
from projet_deepl.tenseurs.tenseur import Tenseur


class NeuralNet:
    def __init__(self, reseau: Sequence[Layer]):
        self.reseau = reseau

    def forward(self, inputs):
        """
        Fonction forward

        Args:
            inputs  (Tenseur)

        Returns:
            (Tenseur)
        """
        for couche in self.reseau:
            inputs = couche.forward(inputs)
        return inputs

    def backward(self, grad):
        """
        Fonction backward

        Args:
            inputs  (Tenseur)

        Returns:
            la dérivée (Tenseur)
        """
        for couche in reversed(self.reseau):
            grad = couche.backward(grad)
        return grad

    def grad_and_param(self) -> Iterator[Tuple[Tenseur, Tenseur]]:
        """
        Fonction grad_and_param

        Args:

        Returns:
            grad et params ( Iterator[Tuple[Tenseur,Tenseur]] )
        """
        for couche in self.reseau:
            for name, param in couche.params.items():
                grad = couche.grads[name]
                yield param, grad
