import matplotlib.pyplot as plt
import sys, os

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)

import numpy as np
from projet_deepl.layers.lineaire import Lineaire
from projet_deepl.lossfunction.mse import MSE
from projet_deepl.layers.sigactivation import SigActivation, sig, sig_prime
from projet_deepl.nn.neuralnet import NeuralNet
from projet_deepl.donnees.donnees import BatchIterateur
from projet_deepl.training.training import Training


from projet_deepl.lossfunction.mae import MAE
from projet_deepl.Optimisation.sgd import SGD


# Donnees
from sklearn import datasets
data = datasets.load_diabetes()

# inputs = np.array([[1, 5, 6], [3, 2, 1], [0, 5, 11], [3, 4, 1]])
# target = np.array([[0], [1], [0], [1]])
import pandas as pd

inputs = data["data"]
target = data["target"]
# Reseau de neurones
nn = NeuralNet([Lineaire(10, 1), SigActivation(sig, sig_prime)])
# Creer les batches

batch = BatchIterateur(1)

import argparse
parser = argparse.ArgumentParser(description='Les hyperparametres')
parser.add_argument('lr', type=float, help='Le learning rate')
parser.add_argument('epochs', type=int, help='Epochs')

args = parser.parse_args()

if __name__ == '__main__':
    Trainer = Training(args.lr, args.epochs)

    errors = Trainer.train(inputs, target, batch, nn, loss=MSE(), optim=SGD())


    for x, y in zip(inputs, target):
        print(x, y)
        predicted = nn.forward(x)
        print(x, predicted)

    x = [[-0.04547248, -0.04464164, -0.0730303,  -0.08141377,  0.08374012,  0.02780893, 0.17381578, -0.03949338, -0.00421986,  0.00306441],
         [-0.00914709, -0.04464164, -0.05686312, -0.05042793,  0.02182224,  0.04534524, -0.02867429,  0.03430886, -0.00991896, -0.01764613],
         [-0.07816532,  0.05068012,  0.07786339,  0.05285819,  0.07823631,  0.0644473, 0.02655027, -0.00259226,  0.04067226, -0.00936191]]
    y = [[1], [183], [233]]

    for i, j in zip(x, y):
        print(i, j)
        predicted = nn.forward(i)
        print(i, predicted)

    plt.plot(errors, color="blue")
    plt.show()

