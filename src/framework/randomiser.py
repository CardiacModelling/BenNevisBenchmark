import numpy as np
import nevis


class Randomiser:
    @staticmethod
    def get_rand_seed(index: int) -> int:
        """
        Get the random seed for the index-th run.

        Parameters
        ----------
        index : int
            The index of the run.
        
        Returns
        -------
        int
            The random seed for the index-th run.
        """
        return index * 1033 + 1234
    
    @staticmethod
    def get_init_guess(index: int) -> np.ndarray:
        """
        Get the initial guess for the index-th run. The 0-th run is a fixed
        point near Ben Nevis.

        Parameters
        ----------
        index : int
            The index of the run.
        
        Returns
        -------
        np.ndarray
            The initial guess for the index-th run.
        """
        if index == 0:
            x, y = nevis.ben().grid
            return np.array([x + 200, y - 200])
        else:
            current_seed = index * 1989 + 2923
            np.random.seed(current_seed)
            x_max, y_max = nevis.dimensions()
            x = np.random.rand() * x_max
            y = np.random.rand() * y_max
            np.random.seed(None)
            return np.array([x, y])