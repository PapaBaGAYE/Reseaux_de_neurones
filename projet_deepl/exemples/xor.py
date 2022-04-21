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


from projet_deepl.layers.tanactivation import TanActivation, tanh, tanh_prime
from projet_deepl.lossfunction.logisticloss import LogisticLoss
from projet_deepl.Optimisation.sgd import SGD


inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

target = np.array([[1], [0], [0], [1]])

nn = NeuralNet([Lineaire(input_size=2, output_size=1), TanActivation(tanh, tanh_prime)])

batch = BatchIterateur(1, True)


Trainer = Training(lr=0.0001)

errors = Trainer.train(inputs, target, batch, nn, loss=MSE(), optim=SGD())


for x, y in zip(inputs, target):
    print(x, y)
    predicted = nn.forward(x)
    print(x, predicted)


plt.plot(errors)
plt.show()
