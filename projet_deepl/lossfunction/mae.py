"""La perte absolue est la moyenne des valeurs absolues des différences entre les valeurs réelles
et les valeurs prédites. 

.. math::
    \\frac{1}{m}\sum_{i = 1}^m\left|y_i - f(x_i)\\right|
"""
from projet_deepl.lossfunction.loss import Loss
from projet_deepl.tenseurs.tenseur import Tenseur, np


class MAE(Loss):
    """
    Classe calcul de la fonction de perte.
    Elle est enfant de la classe Loss.
    """

    def __init__(self):
        pass

    def loss(self, predicted: Tenseur, actual: Tenseur) -> float:
        """
        Determine le gradient de perte

        Args:
            predicted (Tenseur) : Les valeurs prédites
            actual (Tenseur) : Les vraies valeurs

        Returns:
            fonction de perte type (float)
        """
        m = actual.shape[1]
        return (1 / m) * np.nansum(np.abs(actual - predicted))

    def grad_loss(self, predicted: Tenseur, actual: Tenseur) -> Tenseur:
        """
        Calcul du gradient de la fonction de perte

        Args:
            predicted (Tenseur) : Les valeurs prédites
            actual (Tenseur) : Les valeurs réelles

        Returns:
            Le gradient de la fonction de perte (Tenseur)
        """
        m = actual.shape[1]
        ε = 1e-10
        return -(1 / m) * ((actual - predicted) / (np.abs(actual - predicted) + ε))
