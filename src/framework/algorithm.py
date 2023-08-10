from .result import Result
from .algorithm_instance import AlgorithmInstance
from .save_handler import SaveHandler
from .config import RS_ITER_NUM
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import seaborn


class Algorithm:
    def __init__(self, name, func, param_space, version=1):
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
        self.param_space = param_space
        self.version = version
        self.best_instance = None

        self.save_handler = SaveHandler(self)

        self.instances = set()

        # self.load_instances()

    def __call__(self, **params) -> Result:
        return self.func(**params)

    def generate_instance(self, instance_hash=None, **params):
        """
        Generate an ``AlgorithmInstance``.

        Parameters
        ----------
        instance_hash : int
            The hash of the instance. If not provided, a hash will be
            generated based on the time.
        **params
            Hyper-paramters of this instance.
        """
        return AlgorithmInstance(self, params, self.save_handler,
                                 hash=instance_hash)

    def generate_all_instances(self):
        self.instances = set()
        keys = list(self.param_space.keys())
        values = list(self.param_space.values())

        def helper(i, cur_params):
            if i == -1:
                self.instances.add(AlgorithmInstance(
                    self,
                    cur_params,
                    self.save_handler
                ))
                print(f'Instance generated with params {cur_params}')
                return

            for value in values[i]:
                copied_params = dict(cur_params)
                copied_params[keys[i]] = value
                helper(i - 1, copied_params)

        helper(len(keys) - 1, {})

    def generate_random_instance(self):
        """
        Generate a random instance of this algorithm by drawing from the
        hyper-parameter space.
        """
        params = {}
        for param, values in self.param_space.items():
            params[param] = np.random.choice(values)
        return AlgorithmInstance(self, params, self.save_handler)

    def load_instances(self):
        """
        Load all instances from the save folder.
        """
        print('Loading instances...')
        instances = self.save_handler.get_all_instances()
        self.instances.update(instances)
        print(f'{len(instances)} instances loaded.')

    def load_instance(self, instance_hash):
        """
        Load an instance.

        Parameters
        ----------
        instance_hash : float
            The hash of the instance to be loaded.
        """
        instance = self.save_handler.load_instance(instance_hash)
        self.instances.add(instance)

    def tune_params(
        self,
        iter_num=RS_ITER_NUM,
        measure='ert',
        mode='min',
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

        instances = list(self.instances)

        while len(instances) < iter_num:
            new_instance = self.generate_random_instance()
            instances.append(new_instance)

        best_value = float('-inf') if mode == 'max' else float('inf')

        for i, current_instance in enumerate(instances[:iter_num], 1):
            print(f'Calculating instance {i} / {iter_num}')
            current_value = current_instance.performance_measures()[measure]
            print(f'{measure} = {current_value}')
            if (mode == 'max' and current_value >= best_value)\
                    or (mode == 'min' and current_value <= best_value):
                best_value = current_value
                if self.best_instance is not None:
                    self.best_instance.make_results_partial()
                self.best_instance = current_instance
            else:
                current_instance.make_results_partial()

            print()

        self.instances.update(instances)

        return self.best_instance

    def plot_two_measures(self,
                          x_measure='avg_success_eval',
                          y_measure='failure_rate'):
        """
        Plot one performance measure against another on a scatter plot, arcoss
        all instances.

        Parameters
        ----------
        x_measure : string
            The name of the performance measure shown on the x axis.
        y_measure : string
            The name of the performance measure shown on the x axis.
        """
        instances = list(self.instances)
        xs = []
        ys = []

        for instance in instances:
            measures = instance.performance_measures()
            xs.append(measures[x_measure])
            ys.append(measures[y_measure])

        plt.scatter(xs, ys)
        plt.xlabel(x_measure)
        plt.ylabel(y_measure)
        plt.title(f'Performance measures of {self.name}'
                  f'across {len(instances)} instances')
        plt.show()

    def plot_all_measures(self):
        """
        Plot the pair plot of all performance measures, across all instances.
        """
        instances = list(self.instances)
        df = pd.DataFrame([instance.performance_measures()
                          for instance in instances])
        df.drop(['failure_rate', 'success_cnt'], axis=1, inplace=True)
        seaborn.pairplot(df)
        plt.show()

    def plot_tuning(
        self,
        param_x,
        param_y,
        measure_color,
        measure_area,
        x_log=False,
        y_log=False,
        reverse_area=False
    ):
        """
        Plot the performance of all instances generated by hyper-parameter
        tuning, by showing a scatter plot with one hyper-parameter on each
        axis and the color or area of the marks representing designated
        performance measures.

        Parameters
        ----------
        param_x : string
            The name of the hyper-parameter shown on the x axis.
        param_y : string
            The name of the hyper-parameter shown on the y axis.
        measure_color : string
            The name of the performance measure shown using the color of the
            marks.
        measure_area : string
            The name of the performance measure shown using the area of the
            marks.
        x_log : bool
            If the x axis needs to be plotted on the log scale.
        y_log : bool
            If the y axis needs to be plotted on the log scale.
        reverse_area : bool
            If True, a larger area indicates a smaller value of
            ``measure_area``.
        """

        instances = list(self.instances)

        m1s = []
        m2s = []
        xs = []
        ys = []
        for instance in instances:
            xs.append(instance.params[param_x])
            ys.append(instance.params[param_y])
            measures = instance.performance_measures()
            m1s.append(measures[measure_color])
            m2s.append(measures[measure_area])

        if x_log:
            plt.xscale('log')
        if y_log:
            plt.yscale('log')

        m2s = np.array(m2s)
        area_normalizer = matplotlib.colors.Normalize(m2s.min(), m2s.max())
        m2s = area_normalizer(m2s)
        if reverse_area:
            m2s = 1 - m2s

        plt.scatter(xs, ys, s=m2s * 200, c=m1s, alpha=.5)
        plt.colorbar(label=measure_color)
        plt.title('Hyper-parameter tuning results')
        plt.show()
