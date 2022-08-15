from functools import cache
import nevis
import numpy as np
import pickle
import time
from scipy.optimize import dual_annealing


f = nevis.linear_interpolant()
ben_x, ben_y = nevis.ben().grid
def dist_to_ben(x, y):
    return np.linalg.norm(np.array([x, y]) - np.array([ben_x, ben_y]))
x_max, y_max = nevis.dimensions()
MAX_FES = 50000
SUCCESS_HEIGHT = 1340
RS_ITER_NUM = 20
RUN_NUM = 15
SAVE_PATH = './saved_results.pickle'


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
        return int(self.time * 1000)

class SaveHandler:
    def __init__(self):
        try:
            with open(SAVE_PATH, 'rb') as file_obj:
                self.content = pickle.load(file_obj)
        except FileNotFoundError:
            self.content = {}

    def update_algorithm_instance(self, instance_hash, instance):
        h = instance_hash
        if self.content.get(h) is None:
            self.content[h] = instance
        else:
            self.content[h].results.update(instance.results)
        
        with open(SAVE_PATH, 'wb') as file_obj:
            pickle.dump(self.content, file_obj)
    
    def load_algorithm_instance(self, instance_hash):
        return self.content.get(instance_hash)
    
    def get_all_instances(self, algorithm_name=None, algorithm_version=None):
        instances = []
        for instance in self.content.values():
            if (algorithm_name is None or instance.info['algorithm_name'] == algorithm_name)\
                and (algorithm_version is None or instance.info['algorithm_version'] == algorithm_version):
            
                instances.append(instance)
        return instances


class Algorithm:
    def __init__(self, name, func, param_space, version=1):
        self.name = name
        self.func = func
        self.param_space = param_space
        self.version = version
        self.best_instance = None

    def __call__(self, **params) -> Result:
        return self.func(**params)
    

    def generate_random_instance(self):
        params = {}
        for param, values in self.param_space.items():
            params[param] = np.random.choice(values)
        return AlgorithmInstance(self, params)
    
    def tune_params(self, method='random', iter_num=RS_ITER_NUM):
        if method == 'random':

            best_performance = float('inf')
            instances = save_handler.get_all_instances(
                self.name,
                self.version,
            )

            while len(instances) < iter_num:
                new_instance = self.generate_random_instance()
                instances.append(new_instance)

            for current_instance in instances[:iter_num]:
                _, current_performance = current_instance.success_metrics()

                if current_performance < best_performance:
                    best_performance = current_performance
                    self.best_instance = current_instance
            
            print(best_performance)
            
        else:
            raise NotImplementedError    

class AlgorithmInstance:
    def __init__(self, algorithm, params):
        self.algorithm: Algorithm = algorithm
        self.params: dict = params

        self.info = {
            'algorithm_name': algorithm.name,
            'algorithm_version': algorithm.version,
            **params,
        }
        
        self.hash = hash(frozenset(self.info.items()))
        self.results = set()
    
    def load_results(self):
        instance = save_handler.load_algorithm_instance(self.hash)
        if instance is not None:
            self.results.update(instance.results)

    def save_results(self):
        save_handler.update_algorithm_instance(self.hash, self)

    def run(self, run_num):
        self.load_results()

        if len(self.results) >= run_num:
            return

        while len(self.results) < run_num:
            result = self.algorithm(**self.params)
            result.print()
            self.results.add(result)
        print('Saving results...')
        self.save_results()
    
    @cache
    def success_metrics(self,
        max_fes=MAX_FES,
        success_height=SUCCESS_HEIGHT,
        run_num=RUN_NUM
    ):
        self.run(run_num)
        results = list(self.results)[:run_num]
        
        success_cnt = 0
        success_eval_cnt = 0
        for result in results:
            is_success, eval_cnt = result.success_eval(max_fes, success_height)
            if is_success:
                success_cnt += 1
                success_eval_cnt += eval_cnt
        
        success_rate = success_cnt / run_num
        success_performance = float('inf') if success_cnt == 0\
            else success_eval_cnt / success_cnt * run_num / success_cnt
        
        return success_rate, success_performance

    
    def plot_convergence_graph(self):
        pass


if __name__ == '__main__':
    def run_dual_annealing(**kwargs):
        points = []
        function_values = []
        def wrapper(u):
            x, y = u
            points.append((x, y))
            z = f(x, y)
            function_values.append(z)
            return -z

        x_max, y_max = nevis.dimensions()
        ret = dual_annealing(
            wrapper, 
            bounds=[(0, x_max), (0, y_max)],
            maxfun=MAX_FES,
            **kwargs
        )

        return Result(
            ret.x,
            -ret.fun,
            points,
            ret.message,
            function_values,
            ret_obj=ret
        )
    save_handler = SaveHandler()

    algo = Algorithm(
        'Dual Annealing', 
        run_dual_annealing,
        {
            'maxiter': [1000, 1500, 2000],
            'initial_temp': np.linspace(1e3, 5e4, 1000),
            'restart_temp_ratio': np.logspace(-5, -1, 100),
            # 'visit': np.linspace(1 + EPS, 3, 1000),
            # 'accept': np.logspace(-5, -1e-4, 1000),
        }    
    )
    algo.tune_params()
    algo_instance = algo.best_instance
    print(algo_instance.success_metrics())

