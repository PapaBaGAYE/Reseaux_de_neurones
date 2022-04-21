import matplotlib.pyplot as plt
from projet_deepl.donnees.donnees import BatchIterateur
from projet_deepl.nn.neuralnet import NeuralNet
from projet_deepl.Optimisation.optimize import Optimiseur
from projet_deepl.Optimisation.sgd import SGD
from projet_deepl.lossfunction.loss import Loss
from projet_deepl.lossfunction.mse import MSE
from projet_deepl.tenseurs.tenseur import Tenseur
from typing import List


class Training:
    def __init__(self, lr: float = 0.01, epochs: int = 50) -> None:
        """Initialisation des hyperparamètres d'entraînement.

        Args:
            lr (float) : le taux d'apprentissage du réseau
            epochs (int):  le nombre d'itération du réseau
        """
        self.lr: float = lr
        self.epochs: int = epochs

    def train(self, inputs, target, batch_data, nn, loss, optim):
        """Entrainement du réseau

        Args:
            Les données d'entrées inputs : (Tenseur)
            La target target (Tenseur) :
            batchs : (BatchIterator)
            nn : (NeuralNetwork)
            La fonction de perte loss : (MSE)
            La technique d'optisation optim : (SGD)
        """
        errors = []
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for batch in batch_data(inputs, target):
                predicted = nn.forward(batch.inputs)
                epoch_loss += loss.loss(predicted, batch.target)
                grad = loss.gradLoss(predicted, batch.target)
                # print(grad)
                grad = nn.backward(grad)
                # print(grad)
                optim.step(nn)
            errors.append(epoch_loss)

            print(f"Erreur à l'epoch {epoch} est {epoch_loss}")
        return errors
