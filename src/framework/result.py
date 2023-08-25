# from functools import cache
import nevis
import numpy as np
from .config import MAX_FES, SUCCESS_HEIGHT
import matplotlib.pyplot as plt

f = nevis.linear_interpolant()
ben_x, ben_y = nevis.ben().grid


def _dist_to_ben(x, y):
    """Calculate the distance from a point to Ben Nevis."""
    return np.linalg.norm(np.array([x, y]) - np.array([ben_x, ben_y]))


x_max, y_max = nevis.dimensions()


class Result:
    def __init__(self,
                 ret_point,
                 ret_height,
                 run_index,
                 points=[],
                 message='',
                 heights=None,
                 trajectory=[],
                 is_success=None,
                 eval_num=None,
                 len_points=None,
                 ):
        """
        Class for the result of an optimization run.

        Parameters
        ----------
        ret_point : tuple
            The best point (x, y) returned.
        ret_height : float
            The height returned.
        points : array of tuple
            All points visited during the run.
        message : string
            A message describes why the algorithm terminated.
        heights : array of float
            The corresponding function values of ``points``.
        trajectory : array of tuple
            Trajectory used in plots.
        """
        self.ret_point = ret_point
        self.ret_height = ret_height
        self.points = np.array(points)
        if heights is None:
            self.heights = self.get_heights()
        else:
            self.heights = heights

        self.trajectory = np.array(trajectory)
        self.message = message

        if self.points.size == 0:
            assert is_success is not None,\
                'A partial result must have `is_success` and `eval_num`'

        if is_success is None:
            self.is_success, self.eval_num = self.success_eval()
        else:
            self.is_success, self.eval_num = is_success, eval_num
        
        self.run_index = run_index

        if len_points is not None:
            self.len_points = len_points
        else:
            self.len_points = len(self.points)

    def get_heights(self):
        """Calcuate heights for all visited points."""
        return np.array([f(*p) for p in self.points])

    def get_distances(self):
        """Calcuate distances to Ben Nevis for all visited points."""
        return np.array([_dist_to_ben(*p) for p in self.points])

    def success_eval(self):
        """Return a tuple, (is_success, eval_num), indicating if the result is
        succeessful and how many function evaluations it used."""
        for i, h in enumerate(self.heights, 1):
            if i > MAX_FES:
                break

            if h >= SUCCESS_HEIGHT:
                return True, i

        return False, min(MAX_FES, len(self.heights))

    def to_dict(self):
        def to_float_tuple(t):
            x, y = t
            return float(x), float(y)

        points = [to_float_tuple(point) for point in self.points]
        trajectory = [to_float_tuple(point) for point in self.trajectory]
        heights = [float(z) for z in self.heights]

        is_success, eval_num = self.success_eval()

        return {
            'points': points,
            'trajectory': trajectory,
            'heights': heights,

            'ret_point': to_float_tuple(self.ret_point),
            'ret_height': float(self.ret_height),
            'message': self.message,
            'create_time': self.create_time,

            'is_success': is_success,
            'eval_num': eval_num,
            'len_points': len(points),
            'max_fes': MAX_FES,
            'success_height': SUCCESS_HEIGHT,
        }

    def turn_partial(self):
        self.points = np.array([])
        self.heights = np.array([])
        self.distances = np.array([])

    @property
    def ret_distance(self):
        """The distance of result point to Ben Nevis."""
        x, y = self.ret_point
        return _dist_to_ben(x, y)

    def print(self):
        """Print a summary of the result."""
        print(self.create_time)
        print(f'Number of function evals: {len(self.points)}')
        x, y = self.ret_point
        nevis.print_result(x, y, self.ret_height)

    def __eq__(self, other) -> bool:
        return self.index == other.index

    def __hash__(self) -> int:
        return self.index

    @property
    def _plot_labels(self):
        x, y = self.ret_point
        c = nevis.Coords(gridx=x, gridy=y)
        hill, _ = nevis.Hill.nearest(c)

        labels = {
            'Ben Nevis': nevis.ben(),
            hill.name: hill.coords,
            'You': c,
        }

        return labels

    def plot_global(self):
        """Make a 2d global plot of the run process."""
        nevis.plot(
            labels=self._plot_labels,
            points=np.array(self.points),
        )
        plt.show()

    def plot_partial(self, side_length=40e3, zoom=1):
        """
        Make a 2d partial plot of the run process.]

        Parameters
        ----------
        side_length : float
            The side length of the partial plot, in metres.
        zoom : float
            The ratio of zooming used in the plot.
        """
        b = side_length / 2
        x, y = self.ret_point
        boundaries = [x - b, x + b, y - b, y + b]
        nevis.plot(
            boundaries=boundaries,
            labels=self._plot_labels,
            points=self.points,
            zoom=zoom,
        )
        plt.show()

    def generate_kml(self, path):
        """
        Generate a kml file of this run.

        Parameters
        ----------
        path : string
            Path of the generated kml file.
        """
        nevis.generate_kml(
            path,
            labels=self._plot_labels,
            points=self.points
        )
