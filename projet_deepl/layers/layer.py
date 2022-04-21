"""Un reseau de neurones est composé de couches
Exemple: Lineaire->Relu->Lineaire->Relu->lineaire->Sigmoid
     
"""
from projet_deepl.tenseurs.tenseur import Tenseur
from typing import Dict


class Layer:
    """Une couche du réseau de neurone"""

    def __init__(self, intput_size, output_size):
        self.params: Dict[str, Tenseur] = {}
        self.grads: Dict[str, Tenseur] = {}

    def forward(self, inputs: Tenseur) -> Tenseur:
        raise NotImplementedError

    def backward(self, grad: Tenseur):
        raise NotImplementedError
