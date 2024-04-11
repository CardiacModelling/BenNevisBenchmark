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
        return index

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

        np.random.seed(index)
        x_max, y_max = nevis.dimensions()
        x = np.random.rand() * x_max
        y = np.random.rand() * y_max
        np.random.seed(None)
        return np.array([x, y])
