from projet_deepl.nn.neuralnet import NeuralNet


class Optimiseur:
    def __init__(self, lr: float = 0.01) -> None:
        self.lr = lr

    def step(self, reseau: NeuralNet):
        raise NotImplementedError
