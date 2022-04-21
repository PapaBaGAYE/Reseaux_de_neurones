from projet_deepl.Optimisation.optimize import Optimiseur
from projet_deepl.nn.neuralnet import NeuralNet


class SGD(Optimiseur):
    def __init__(self, lr: float = 0.001) -> None:
        super().__init__(lr)

    def step(self, reseau: NeuralNet):
        """
        Determine du pas

        Args:
            reseau (NeuralNet) : Les réseaux de neurones

        Returns:

        """
        for param, grad in reseau.grad_and_param():
            param -= self.lr * grad
