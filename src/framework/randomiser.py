import numpy as np
import nevis


class Randomiser:
    @staticmethod
    def get_rand_seed(index: int) -> int:
        return index * 1033 + 1234
    
    @staticmethod
    def get_init_guess(index: int) -> np.ndarray:
        current_seed = index * 1989 + 2923
        np.random.seed(current_seed)
        x_max, y_max = nevis.dimensions()
        x = np.random.rand() * x_max
        y = np.random.rand() * y_max
        np.random.seed(None)
        return np.array([x, y])