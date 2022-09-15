from functools import cache
import nevis
import numpy as np
import time
from .config import MAX_FES, SUCCESS_HEIGHT
import nevis
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
        points, 
        message='',
        heights=None, 
        distances=None,
        trajectory=[],
        ret_obj=None
    ):
        # returned point, (x, y)
        self.ret_point = ret_point
        # returned height
        self.ret_height = ret_height
        # all points at which the function was evaluated
        self.points = np.array(points)
        # corresponding function values
        if heights is None:
            self.heights = self.get_heights()
        else:
            self.heights = heights
        # corresponding distances to Ben Neivs
        if distances is None:
            self.distances = self.get_distances()
        else:
            self.distances = distances
        # optional trajectory, used in plots
        self.trajectory = np.array(trajectory)
        # a message describes why the algorithm terminated
        self.message = message
        # optional, result object returned by the optimization process
        self.ret_obj = ret_obj
        # used as an identifier
        self.time = time.time()

    def get_heights(self):
        return np.array([f(*p) for p in self.points])
    
    def get_distances(self):
        return np.array([_dist_to_ben(*p) for p in self.points])
    
    def success_eval(self, max_fes=MAX_FES, success_height=SUCCESS_HEIGHT):
        """Return a tuple, (is_success, eval_num), indicating if the result is succeessful
        and how many function evaluations it used."""
        for i, h in enumerate(self.heights, 1):
            if i > max_fes:
                break
                
            if h >= success_height:
                return True, i
        
        return False, min(max_fes, len(self.heights))
    
    @property
    def ret_distance(self):
        x, y = self.ret_point
        return _dist_to_ben(x, y)
    
    def print(self):
        x, y = self.ret_point
        nevis.print_result(x, y, self.ret_height)
    
    def __eq__(self, other) -> bool:
        return self.time == other.time
    
    def __hash__(self) -> int:
        return int(self.time * 1000000)
    
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
        nevis.plot(
            labels=self._plot_labels,
            points=np.array(self.points),
        )
        plt.show()
    
    def plot_partial(self, side_length=40e3, zoom=1):
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
        nevis.generate_kml(
            path, 
            labels=self._plot_labels,
            points=self.points
        )
