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
    def __init__(
        self,
        ret_point,
        ret_height,
        points=[],
        message='',
        heights=None,
        trajectory=[],
        is_success=None,
        eval_num=None,
        len_points=None,
        info=None,
        algorithm_name=None,
        algorithm_version=None,
        instance_index=None,
        result_index=None,
    ):
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

        if len_points is not None:
            self.len_points = len_points
        else:
            self.len_points = len(self.points)

        self.info = {
            'algorithm_name': algorithm_name,
            'algorithm_version': algorithm_version,
            'instance_index': instance_index,
            'result_index': result_index,
        }
        if None in self.info.values() and info is not None: 
            self.info = info

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
        def to_float_list(t):
            x, y = t
            return [float(x), float(y)]
        return {
            **self.info,
            # 'algorithm_name': self.info['algorithm_name'],
            # 'algorithm_version': self.info['algorithm_version'],
            # 'instance_index': self.info['instance_index'],
            # 'result_index': self.info['result_index'],

            'ret_point': to_float_list(self.ret_point),
            'ret_height': float(self.ret_height),
            'message': self.message,

            'is_success': self.is_success,
            'eval_num': self.eval_num,
            'len_points': self.len_points,

            'info': self.info,
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
        print(f'Number of function evals: {len(self.points)}')
        x, y = self.ret_point
        nevis.print_result(x, y, self.ret_height)

    def __eq__(self, other) -> bool:
        return self.info == other.info

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
