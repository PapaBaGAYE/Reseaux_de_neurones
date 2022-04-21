import numpy as np
from projet_deepl.tenseurs.tenseur import Tenseur
from projet_deepl.layers.layer import Layer
from projet_deepl.initialisation.initialisation import Initialiser
from typing import Dict


class Lineaire(Layer):
    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__(input_size, output_size)
        self.input_size = input_size
        self.output_size = output_size
        # self.params["W"] = np.zeros((input_size, output_size))
        # self.params["b"] = np.zeros(output_size)

        self.params["W"] = Initialiser(
            self.input_size, self.output_size
        ).initialisation_He()
        self.params["b"] = Initialiser(self.input_size, self.output_size).init_zeros()

    def forward(self, inputs: Tenseur) -> Tenseur:
        """
        Fonction forward

        Args:
            inputs (Tenseur)

        Returns:
            Sortie de la couche (Tenseur)
        """
        self.inputs = inputs
        return np.dot(inputs, self.params["W"]) + self.params["b"]

    def backward(self, grad: Tenseur) -> Tenseur:
        """
        Fonction backward

        Args:
            inputs (Tenseur)

        Returns:
            (Tenseur)
        """
        self.grads["W"] = np.dot(self.inputs.T, grad)
        self.grads["b"] = np.sum(grad, axis=0)
        return np.dot(self.params["W"], grad)
