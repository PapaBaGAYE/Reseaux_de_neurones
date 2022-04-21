import numpy as np
from projet_deepl.tenseurs.tenseur import Tenseur


class Loss:
    def __init__(self):
        raise NotImplementedError

    def loss(self, predicted: Tenseur, actual: Tenseur) -> float:
        raise NotImplementedError

    def gradLoss(self, predicted: Tenseur, actual: Tenseur) -> Tenseur:
        raise NotImplementedError
