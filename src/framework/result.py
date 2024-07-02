# from functools import cache
import nevis
import numpy as np
from .config import MAX_FES, SUCCESS_HEIGHT, GARY_SCORE_CONFIG
import matplotlib.pyplot as plt

f = nevis.linear_interpolant()
ben_x, ben_y = nevis.ben().grid


def _dist_to_ben(x, y):
    """Calculate the distance from a point to Ben Nevis."""
    return np.linalg.norm(np.array([x, y]) - np.array([ben_x, ben_y]))


x_max, y_max = nevis.dimensions()


class Result:
    """
    Class for a result of an algorithm instance.

    Attributes
    ----------
    ret_point : tuple
        The returned point of the result.
    ret_height : float
        The returned height of the result.
    points : list
        The list of all visited points.
    message : string
        The message of the result.
    heights : list
        The list of all visited heights.
    trajectory : list
        A self defined list of points that the algorithm has visited.
    is_success : bool
        Whether the result is successful. For loading saved results.
    gary_score : int
        The Gary score of the result. For loading saved results.
    eval_num : int
        The number of function evaluations used. For loading saved results.
    len_points : int
        The number of points visited. For loading saved results.
    info : dict
        The information of the result.
    algorithm_name : string
        The name of the algorithm.
    algorithm_version : int
        The version of the algorithm.
    instance_index : int
        The index of the instance.
    result_index : int
        The index of the result.
    end_of_iterations : list of ints
        The indices of function evaluations which mark end of each of the iterations
        within the run. Used in making animations.

    Methods
    -------
    set_info(info, algorithm_name, algorithm_version, instance_index, result_index)
        Set the information of the result.
    get_heights()
        Calcuate heights for all visited points.
    success_eval()
        Return a tuple, (is_success, eval_num, gary_score), indicating if the result is
        succeessful and how many function evaluations it used.
    to_dict(partial=True)
        Return a dictionary representation of the result.
    print()
        Print a summary of the result.
    plot_global()
        Make a 2d global plot of the run process.
    plot_local(side_length=40e3, zoom=1)
        Make a 2d local plot of the run process.
    generate_kml(path)
        Generate a kml file of this run.
    """

    def __init__(
        self,
        ret_point,
        ret_height,
        points=[],
        message='',
        heights=None,
        trajectory=[],
        is_success=None,
        gary_score=None,
        eval_num=None,
        len_points=None,
        info=None,
        algorithm_name=None,
        algorithm_version=None,
        instance_index=None,
        result_index=None,
        end_of_iterations=[],
    ):
        self.ret_point = ret_point
        self.ret_height = ret_height
        self.points = np.array(points)

        if len_points is not None:
            self.len_points = len_points
        else:
            self.len_points = len(self.points)

        if heights is None:
            self.heights = self.get_heights()
        else:
            self.heights = heights

        self.trajectory = np.array(trajectory)
        self.message = message

        if self.points.size == 0:
            assert is_success is not None, \
                'A partial result must have `is_success` and `eval_num`'

        if is_success is None:
            self.is_success, self.eval_num, self.gary_score = self.success_eval()
        else:
            self.is_success, self.eval_num, self.gary_score = is_success, eval_num, gary_score

        self.set_info(info, algorithm_name, algorithm_version,
                      instance_index, result_index)

        if end_of_iterations == []:
            # if there is no end_of_iterations given,
            # then we consider the whole run as a single iteration
            self.end_of_iterations = [self.len_points]
        else:
            self.end_of_iterations = end_of_iterations

    def set_info(
        self,
        info=None,
        algorithm_name=None,
        algorithm_version=None,
        instance_index=None,
        result_index=None,
    ):
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

    @property
    def distances(self):
        """Calcuate distances to Ben Nevis for all visited points."""
        return np.array([_dist_to_ben(*p) for p in self.points])

    def success_eval(self):
        """Return a tuple, (is_success, eval_num, gary_score), indicating if the result is
        succeessful and how many function evaluations it used."""

        # if self.ret_height < SUCCESS_HEIGHT:
        #     return False, min(MAX_FES, self.len_points)

        max_height = 0
        for i, h in enumerate(self.heights, 1):
            if i > MAX_FES:
                break
            max_height = max(h, max_height)
            if h >= SUCCESS_HEIGHT:
                return True, i, 10

        gary_score = 0
        for g_h, g_p in GARY_SCORE_CONFIG:
            if max_height >= g_h:
                gary_score = g_p
                break

        return False, min(MAX_FES, self.len_points), gary_score

    def __repr__(self) -> str:
        return str(self.to_dict())

    def to_dict(self, partial=True):
        def to_float_list(t):
            x, y = t
            return [float(x), float(y)]
        res = {
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
            'gary_score': self.gary_score,

            'info': self.info,
        }

        if not partial:
            res['points'] = [to_float_list(p) for p in self.points]
            res['end_of_iterations'] = list(int(i) for i in self.end_of_iterations)

        return res

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

    def plot_local(self, side_length=40e3, zoom=1):
        """
        Make a 2d local plot of the run process.]

        Parameters
        ----------
        side_length : float
            The side length of the local plot, in metres.
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
