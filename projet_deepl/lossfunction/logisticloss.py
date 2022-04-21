from projet_deepl.tenseurs.tenseur import Tenseur, np
from projet_deepl.lossfunction.loss import Loss


class LogisticLoss(Loss):
    """Classe calcul de la fonction de perte.
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
            fonction de perte (float)
        """
        m = len(predicted)
        return (1 / m) * np.nansum(
            -np.multiply(actual, np.log(predicted))
            - np.multiply(1 - actual, np.log(1 - predicted))
        )

    def gradLoss(self, predicted: Tenseur, actual: Tenseur) -> Tenseur:
        """
        Fonction de perte

        Args:
            predicted (Tenseur) : Les valeurs prédites
            actual (Tenseur) : Les valeurs réelles

        Returns:
            la dérivée (Tenseur)
        """
        m = len(predicted)
        return (1 / m) * (
            -np.divide(actual, predicted) + np.divide((1 - actual), (1 - predicted))
        )
