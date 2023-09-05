from .algorithm_instance import AlgorithmInstance
from .config import RS_ITER_NUM, MAX_INSTANCE_FES, logger
import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib
# import pandas as pd
# import seaborn


class Algorithm:
    def __init__(self, 
                 name: str, 
                 func, 
                 param_space: dict, 
                 version: int=1):
        """
        Class for an algorithm.

        Parameters
        ----------
        name : string
            The name of the algorithm.
        func : function
            The function that runs the algorithm. This function takes all the
            hyper-paramters used by the algorithm as keyword arguments, and
            returns a ``Result`` object.
        param_space : dict
            A dictionary mapping the names of hyper-paramters to the list of
            values they can take.
        version : int
            The version of the algorithm. This is used to distinguish between
            different versions of the same algorithm so that they are saved in
            different folders.
        """
        self.name = name
        self.func = func
        self.version = version
        self.best_instance = None
        self.best_instance_index = -2

        self.param_space = param_space
        self.param_keys = list(param_space.keys())
        self.param_values = list(param_space.values())
        self.param_value_lens = [len(values) for values in self.param_values]
        self.param_space_size = np.prod(self.param_value_lens)

        self.instance_indices = set()
    
    @property
    def info(self):
        return {
            'algorithm_name': self.name,
            'algorithm_version': self.version,
        }
    
    def index_to_tuple(self, index):
        assert index != -1

        result = []
        for l in self.param_value_lens[::-1]:
            result.append(index % l)
            index //= l
        return tuple(result[::-1])

    def index_to_params(self, index):
        if index == -1:
            return {}
        tpl = self.index_to_tuple(index)
        return {k: self.param_space[k][i] 
                for k, i in zip(self.param_keys, tpl)}
    
    def tuple_to_index(self, t):
        result = 0
        for i, l in enumerate(self.param_value_lens):
            result *= l
            result += t[i]
        return int(result)
    
    def params_to_index(self, params):
        """Find the closest instance index to the given params."""
        t = []
        for k in self.param_keys:
            assert k in params.keys(), f'Key {k} not in params'
            vs = np.array(self.param_space[k])
            v = params[k]
            i = np.argmin(np.abs(vs - v))
            t.append(i)
        return self.tuple_to_index(tuple(t))
    
    def generate_default_instance(self):
        return self.generate_instance(-1)

    def generate_instance(self, instance_index): 
        """
        Generate an ``AlgorithmInstance``.

        Parameters
        ----------
        instance_index : int
            The index of the instance to be generated.
        """
        instance = AlgorithmInstance(self, instance_index)
        self.instance_indices.add(instance_index)
        return instance

    def generate_instance_from_params(self, **params) -> AlgorithmInstance:
        """
        Generate an ``AlgorithmInstance``.

        Parameters
        ----------
        **params
            Hyper-paramters of this instance.
        """
        instance_index = self.params_to_index(params)
        return self.generate_instance(instance_index)

    def generate_random_instance(self) -> AlgorithmInstance:
        """
        Generate a random instance of this algorithm by drawing from the
        hyper-parameter space.
        """
        i = -1
        while i in self.instance_indices:
            i = np.random.randint(0, self.param_space_size)
        return self.generate_instance(i)
    
    def save_best_instance(self, save_handler=None):
        if save_handler is not None:
            save_handler.save_algorithm_best_instance(self)
    
    def load_best_instance(self, save_handler=None, result_partial=True):
        if save_handler is not None:
            save_handler.load_algorithm_best_instance(self, result_partial)
    
    def load_instance_indices(self, save_handler=None):
        if save_handler is not None:
            save_handler.load_algorithm_instance_indices(self)

    def tune_params(
        self,
        iter_num=RS_ITER_NUM,
        measure='gary_ert',
        mode='min',
        save_handler=None,
        max_instance_fes=MAX_INSTANCE_FES,
        rand_seed=None,
    ):
        """
        Hyper-paramter tuning using random search.

        Parameters
        ----------
        iter_num : int
            The number of iterations used in random search.
        measure : string
            The name of the performance measure used to select the best
            instance.
        mode : string, 'min' or 'max'
            Whether to select the instance with the minimum/maximum ``measure``
            as the best one.


        Returns
        -------
        An ``AlgorithmInstance``, the best instance found in terms of
        ``measure``.
        """

        if self.best_instance is not None:
            best_value = self.best_instance.performance_measures()[measure]
        else:
            best_value = float('-inf') if mode == 'max' else float('inf')

        n = min(iter_num, self.param_space_size)
        while len(self.instance_indices) < n:
            logger.info(f'{len(self.instance_indices)} / {n} instances...')
            np.random.seed(rand_seed)
            current_instance = self.generate_random_instance()
            logger.debug(current_instance.params)
            current_instance.run(
                save_handler=save_handler,
                restart=True,
                max_instance_fes=max_instance_fes,
            )
            current_value = current_instance.performance_measures()[measure]
            logger.debug(current_instance.performance_measures())
            logger.info(f'{measure} = {current_value}')
            current_instance.make_results_partial()
            if (mode == 'max' and current_value >= best_value)\
                    or (mode == 'min' and current_value <= best_value):
                best_value = current_value
                self.best_instance = current_instance
                self.best_instance_index = current_instance.instance_index
                self.save_best_instance(save_handler=save_handler)
            logger.info('===')

        logger.info(f'Best instance selected for {self.name}:')
        logger.info(self.best_instance.info)
        logger.info(self.best_instance.params)
        logger.info(self.best_instance.performance_measures(excluding_first=True))
        return self.best_instance

    # def plot_two_measures(self,
    #                       x_measure='avg_success_eval',
    #                       y_measure='failure_rate'):
    #     """
    #     Plot one performance measure against another on a scatter plot, arcoss
    #     all instances.

    #     Parameters
    #     ----------
    #     x_measure : string
    #         The name of the performance measure shown on the x axis.
    #     y_measure : string
    #         The name of the performance measure shown on the x axis.
    #     """
    #     instances = list(self.instances)
    #     xs = []
    #     ys = []

    #     for instance in instances:
    #         measures = instance.performance_measures()
    #         xs.append(measures[x_measure])
    #         ys.append(measures[y_measure])

    #     plt.scatter(xs, ys)
    #     plt.xlabel(x_measure)
    #     plt.ylabel(y_measure)
    #     plt.title(f'Performance measures of {self.name}'
    #               f'across {len(instances)} instances')
    #     plt.show()

    # def plot_all_measures(self):
    #     """
    #     Plot the pair plot of all performance measures, across all instances.
    #     """
    #     instances = list(self.instances)
    #     df = pd.DataFrame([instance.performance_measures()
    #                       for instance in instances])
    #     df.drop(['failure_rate', 'success_cnt'], axis=1, inplace=True)
    #     seaborn.pairplot(df)
    #     plt.show()

    # def plot_tuning(
    #     self,
    #     param_x,
    #     param_y,
    #     measure_color,
    #     measure_area,
    #     x_log=False,
    #     y_log=False,
    #     reverse_area=False
    # ):
    #     """
    #     Plot the performance of all instances generated by hyper-parameter
    #     tuning, by showing a scatter plot with one hyper-parameter on each
    #     axis and the color or area of the marks representing designated
    #     performance measures.

    #     Parameters
    #     ----------
    #     param_x : string
    #         The name of the hyper-parameter shown on the x axis.
    #     param_y : string
    #         The name of the hyper-parameter shown on the y axis.
    #     measure_color : string
    #         The name of the performance measure shown using the color of the
    #         marks.
    #     measure_area : string
    #         The name of the performance measure shown using the area of the
    #         marks.
    #     x_log : bool
    #         If the x axis needs to be plotted on the log scale.
    #     y_log : bool
    #         If the y axis needs to be plotted on the log scale.
    #     reverse_area : bool
    #         If True, a larger area indicates a smaller value of
    #         ``measure_area``.
    #     """

    #     instances = list(self.instances)

    #     m1s = []
    #     m2s = []
    #     xs = []
    #     ys = []
    #     for instance in instances:
    #         xs.append(instance.params[param_x])
    #         ys.append(instance.params[param_y])
    #         measures = instance.performance_measures()
    #         m1s.append(measures[measure_color])
    #         m2s.append(measures[measure_area])

    #     if x_log:
    #         plt.xscale('log')
    #     if y_log:
    #         plt.yscale('log')

    #     m2s = np.array(m2s)
    #     area_normalizer = matplotlib.colors.Normalize(m2s.min(), m2s.max())
    #     m2s = area_normalizer(m2s)
    #     if reverse_area:
    #         m2s = 1 - m2s

    #     plt.scatter(xs, ys, s=m2s * 200, c=m1s, alpha=.5)
    #     plt.colorbar(label=measure_color)
    #     plt.title('Hyper-parameter tuning results')
    #     plt.show()
