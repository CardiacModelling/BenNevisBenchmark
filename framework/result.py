import nevis
import numpy as np
import time
from .config import MAX_FES, SUCCESS_HEIGHT

f = nevis.linear_interpolant()
ben_x, ben_y = nevis.ben().grid
def dist_to_ben(x, y):
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
        ret_obj=None
    ):
        self.ret_point = ret_point
        self.ret_heigt = ret_height
        self.points = np.array(points)
        if heights is None:
            self.heights = self.get_heights()
        else:
            self.heights = heights
        if distances is None:
            self.distances = self.get_distances()
        else:
            self.distances = distances
        
        self.message = message
        self.ret_obj = ret_obj
        self.time = time.time()

    def get_heights(self):
        return np.array([f(*p) for p in self.points])
    
    def get_distances(self):
        return np.array([dist_to_ben(*p) for p in self.points])
    
    def success_eval(self, max_fes=MAX_FES, success_height=SUCCESS_HEIGHT):
        for i, h in enumerate(self.heights, 1):
            if i > max_fes:
                break
                
            if h >= success_height:
                return True, i
        
        return False, max_fes
    
    def print(self):
        x, y = self.ret_point
        nevis.print_result(x, y, self.ret_heigt)
    
    def __eq__(self, other) -> bool:
        return self.time == other.time
    
    def __hash__(self) -> int:
        return int(self.time * 1000000)