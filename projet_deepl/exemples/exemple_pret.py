import matplotlib.pyplot as plt
import sys, os
import pyment

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)

import numpy as np
from projet_deepl.layers.lineaire import Lineaire
from projet_deepl.lossfunction.mse import MSE
from projet_deepl.layers.sigactivation import SigActivation, sig, sig_prime
from projet_deepl.nn.neuralnet import NeuralNet
from projet_deepl.donnees.donnees import BatchIterateur
from projet_deepl.training.training import Training


from projet_deepl.lossfunction.logisticloss import LogisticLoss
from projet_deepl.Optimisation.sgd import SGD


# Donnees
from sklearn import datasets
iris = datasets.load_iris()

# inputs = np.array([[1, 5, 6], [3, 2, 1], [0, 5, 11], [3, 4, 1]])
# target = np.array([[0], [1], [0], [1]])

import pandas as pd

inputs = np.array(pd.read_csv('data.csv'))
target = np.array(pd.read_csv('target.csv'))
# Reseau de neurones
nn = NeuralNet([Lineaire(3, 1), SigActivation(sig, sig_prime)])
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

    x = [[1.0, 0, 0.0], [1.0, 1,1508.0],  [1.0, 1,1508.0]]
    y = [[1], [0], [1]]

    for i, j in zip(x, y):
        print(i, j)
        predicted = nn.forward(i)
        print(i, predicted)

    plt.plot(errors, color="blue")
    plt.show()