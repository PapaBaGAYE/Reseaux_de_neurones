import numpy as np


class Initialiser:
    def __init__(self, input_size: int, output_size: int):
        self.input_size = input_size
        self.output_size = output_size

    def init_zeros(self):
        """
        Fonction d'initialisation zeros
        Args:

        Returns:
            y0 (float)
        """
        y0 = np.random.randn(self.output_size)
        return y0

    def init_aleatoire(self):
        """
        Fonction d'initialisation aléatoires

        Args:

        Returns:
            X0 (Tuple)
        """
        X0 = np.random.randn(self.input_size, self.output_size)
        return X0

    def initialisation_He(self):
        """
        Fonction d'initialisation he

        Args:


        Returns:
            X0 (Tuple)
        """
        X0 = np.random.randn(self.input_size, self.output_size) * np.sqrt(
            2.0 / self.output_size
        )
        return X0
