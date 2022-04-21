import numpy as np
from projet_deepl.lossfunction.loss import Loss
from projet_deepl.tenseurs.tenseur import Tenseur


class MSE(Loss):
    def __init__(self):
        pass

    def loss(self, predicted: Tenseur, actual: Tenseur) -> float:
        """
        Determine le gradient de perte

        Args:
            predicted (Tenseur) : Les valeurs prédites type
            actual (Tenseur) : Les vraies valeurs type

        Returns:
            la fonction de perte (float)
        """
        return (1 / len(actual)) * np.sum((predicted - actual) ** 2)

    def gradLoss(self, predicted: Tenseur, actual: Tenseur) -> Tenseur:
        """
        Fonction de perte

        Args:
            predicted (Tenseur) : Les valeurs prédites type
            actual (Tenseur) : Les valeurs réelles type

        Returns:
            la dérivée (Tenseur)
        """
        return (2 / len(actual)) * (predicted - actual)
