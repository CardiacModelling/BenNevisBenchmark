from textwrap import wrap
import nevis
from functools import wraps

from .result import Result

f = nevis.linear_interpolant()

def optimizer(opt):
    
    @wraps(opt)
    def func(**params):
        points = []
        function_values = []
        def wrapper(u):
            x, y = u
            points.append((x, y))
            z = f(x, y)
            function_values.append(z)
            return -z
        
        x_max, y_max = nevis.dimensions()
        
        res_dict = opt(
            wrapper,
            x_max,
            y_max,
            **params
        )

        x = res_dict['x']
        z = res_dict['z']
        message = res_dict.get('message', '')
        trajectory = res_dict.get('trajectory', [])
        ret_obj = res_dict.get('ret_obj')

        return Result(
            x,
            -z,
            points,
            message,
            function_values,
            trajectory=trajectory,
            ret_obj=ret_obj
        )
    return func