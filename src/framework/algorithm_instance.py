from functools import cache
import pprint
from typing import Any
import nevis
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from .config import MAX_FES, RUN_NUM, MAX_INSTANCE_FES
import time
from tqdm import tqdm
import logging
from .randomiser import Randomiser


def pad_list(ls):
    """Making each list of the input list of lists have the same length, by
    padding with the last element of each list."""
    if not ls:
        return []
    length = max(len(lst) for lst in ls)
    return [
        np.append(lst, [lst[-1]] * (length - len(lst)))
        for lst in ls
    ]


class AlgorithmInstance:
    def __init__(self, algorithm, instance_index):
        """
        Class for an algorithm instance. All results of this instance
        previously saved will be loaded automatically.

        Parameters
        ----------
        algorithm : Algorithm
            The algorithm this instance belongs to.
        """

        self.algorithm = algorithm
        self.instance_index = instance_index
        
        self.results = dict()

    
    def __call__(self, result_index):
        if self.results.get(result_index) is not None:
            return self.results[result_index]

        params = self.algorithm.index_to_params(self.instance_index)
        rand_seed = Randomiser.get_rand_seed(result_index)
        init_guess = Randomiser.get_init_guess(result_index)
        result = self.algorithm.func(rand_seed, init_guess, **params)

        result.info = self.info
        result.info['result_index'] = result_index

        self.results[result_index] = result

        return result

    
    @property
    def info(self):
        return {
            'algorithm_name': self.algorithm.name,
            'algorithm_version': self.algorithm.version,
            'instance_index': self.instance_index,
        }

    def __eq__(self, other) -> bool:
        return self.info == other.info

    # def __hash__(self):
    #     # return hash(self.algorithm.name + "$" + str(self.algorithm.version) + "$" + str(self.instance_index))

    def run(self, save_handler):
        """
        Run this instance and save all results.
        """

        save_handler.save_instance(self)

        current_instance_fes = 0
        for result in self.results:
            current_instance_fes += result.eval_num
        
        if current_instance_fes >= MAX_INSTANCE_FES:
            return

        logging.info(f'Running instance {self.hash}')
        logging.info(pprint.pformat(self.info))
        logging.debug(f'Current instance fes: {current_instance_fes}')

        while current_instance_fes < MAX_INSTANCE_FES:
            run_index = len(self.results)
            logging.debug(f'Running instance {self.hash} #{run_index}')

            result = self.algorithm(run_index, self.instance_index)
            self.results.add(result)
            current_instance_fes += result.eval_num

            save_handler.add_result(self, result)

    def make_results_partial(self):
        for result in self.results:
            result.turn_partial()

    def fetch_full_results(self):
        if self.results_patial:
            self.load_results(False)

    def load_results(self, save_handler):
        """Load all results saved for this instance."""
        results = save_handler.load_results(self.hash)
        self.results = results
        if results:
            self.results_patial = True

    def performance_measures(self):
        """
        Return all the performance measures of the instance. It's safe to run this
        method if even the results are ``partial''.

        Returns
        -------
        dict
            A dictionary of performance measures. Keys are the names of the
            measures.
            The keys include:
            - 'success_rate': The success rate of the instance.
            - 'failure_rate': The failure rate of the instance.
            - 'success_cnt': The number of successful runs.
            - 'avg_success_eval': The average number of function evaluations
               for successful runs.
            - 'hv': The hypervolume.
            - 'par2': Penalized average runtime with a factor of 2.
            - 'par10': Penalized average runtime with a factor of 10.
            - 'avg_height': The average height of the returned points.
            - 'ert': The expected runtime.
            - 'sp': The success performance.
        """

        results = list(self.results)
        run_num = len(results)

        success_evals = []
        failed_evals = []
        ret_heights = []
        for result in results:
            is_success, eval_cnt = result.is_success, result.eval_num
            if is_success:
                success_evals.append(eval_cnt)
            else:
                failed_evals.append(eval_cnt)

            ret_heights.append(result.ret_height)

        success_cnt = len(success_evals)
        success_eval_cnt = sum(success_evals)
        failed_cnt = len(failed_evals)

        if success_cnt == 0:
            return {
                'success_rate': 0,
                'failure_rate': 1,
                'success_cnt': 0,
                'avg_success_eval': float('inf'),
                'hv': 0,
                'par2': float('inf'),
                'par10':  float('inf'),
                'avg_height': 0,
                'ert': float('inf'),
                'sp': float('inf'),
                'ert_std': float('inf'),
                'success_rate_upper': 0,
                'success_rate_lower': 0,
                'success_rate_length': 0,
            }

        success_eval_var = np.var(success_evals, ddof=1)
        failed_eval_var = np.var(failed_evals, ddof=1)
        avg_failed_eval_sqr = np.mean(np.square(failed_evals))

        success_rate = success_cnt / run_num
        avg_success_eval = np.mean(success_evals)
        avg_failed_eval = np.mean(failed_evals) if failed_cnt != 0 \
            else 0

        # Wilson score interval for success rate
        z = 1.96
        radius = z / (run_num + z ** 2) * np.sqrt(
            success_cnt * failed_cnt / run_num + z ** 2 / 4)
        center = (success_cnt + 0.5 * z ** 2) / (run_num + z ** 2)

        return {
            'success_rate': success_rate,
            'failure_rate': 1 - success_rate,
            'success_cnt': success_cnt,
            'avg_success_eval': avg_success_eval,
            'hv': (MAX_FES - avg_success_eval) * success_rate,
            'par2': (failed_cnt * 2 * MAX_FES + success_eval_cnt) / run_num,
            'par10':  (failed_cnt * 10 * MAX_FES + success_eval_cnt) / run_num,
            'avg_height': np.mean(ret_heights),
            'ert': avg_success_eval + (
                (1 - success_rate) / success_rate * avg_failed_eval
            ),
            'sp': avg_success_eval / success_rate,
            'ert_std': np.sqrt(
                (1 - success_rate) / success_rate * failed_eval_var + (
                    (1-success_rate) / (success_rate**2) * avg_failed_eval_sqr
                ) + success_eval_var),
            'success_rate_upper': center + radius,
            'success_rate_lower': center - radius,
            'success_rate_length': radius * 2,
        }

    # def plot_measure_by_runs(self, measures=['ert'], max_run_num=RUN_NUM):
    #     self.run(run_num=max_run_num)
    #     ys = []
    #     xs = list(range(1, max_run_num + 1))
    #     for i in xs:
    #         cur_ys = []
    #         for measure in measures:
    #             y = self.performance_measures(run=True, run_num=i)[measure]
    #             cur_ys.append(y)
    #         print(i, cur_ys)
    #         ys.append(cur_ys)
    #     fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    #     fig.suptitle(
    #         f'Measures by number of runs for instance {self.hash}'
    #         ' of algorithm ' + self.info['algorithm_name'])
    #     ax.set_xlabel('Number of runs')
    #     ax.set_ylabel('Measure')
    #     for i, measure in enumerate(measures):
    #         ax.plot(xs, [y[i] for y in ys], label=measure)
    #     ax.legend(loc='upper right')
    #     plt.show()

    def __repr__(self) -> str:
        return pprint.pformat(self.info)

    def print_results(self):
        """Print all results of this instance."""
        self.fetch_full_results()

        pprint(self.performance_measures())
        for i, result in enumerate(self.results):
            print(f'=== Result #{i} ===')
            print(f'{result.len_points}')
            result.print()
            print()

    # def plot_histogram(self):
    #     """Plot the histogram of the heights, distances to Ben Nevis, and
    #     numbers of function of evaluations of all results."""
    #     heights = []
    #     distances = []
    #     evals = []

    #     for result in self.results:
    #         heights.append(result.ret_height)
    #         distances.append(result.ret_distance)
    #         evals.append(result.len_points)

    #     fig, axs = plt.subplots(1, 3, figsize=(18, 10))

    #     axs[0].hist(heights)
    #     axs[1].hist(distances)
    #     axs[2].hist(evals)
    #     fig.suptitle(
    #         f'Histogram of returned heights, distances to Ben Nevis, and'
    #         f'numbers of function evals for {len(self.results)} runs of '
    #         f'{self.algorithm.name}')
    #     plt.show()

    # def plot_ret_points(self):
    #     """
    #     Plot a map of the returned points of all results.
    #     """
    #     ret_points = []

    #     for result in self.results:
    #         ret_points.append(result.ret_point)

    #     nevis.plot(
    #         labels={
    #             'Ben Neivs': nevis.ben(),
    #             'Ben Macdui': nevis.Hill.by_rank(2).coords,
    #         },
    #         points=np.array(ret_points)
    #     )

    #     plt.show()

    def plot_convergence_graph(self, downsampling=1):
        """
        Plot a convergence graph across all instances.

        Parameters
        ----------
        downsampling : int
            Downsampling factor on number of function evaluations.
        """

        self.fetch_full_results()

        def calc(values_list, method):
            random_results = []
            for values in values_list:
                prefix = (np.maximum if method ==
                          'max' else np.minimum).accumulate(values)
                random_results.append(prefix)

            temp = np.array(random_results).T

            re_mean = []
            re_0 = []
            re_25 = []
            re_50 = []
            re_75 = []
            re_100 = []
            for x in temp:
                re_mean.append(np.mean(x))
                sorted_x = np.sort(x)
                re_0.append(sorted_x[0])
                re_25.append(sorted_x[int(len(sorted_x) * 0.25)])
                re_50.append(sorted_x[int(len(sorted_x) * 0.50)])
                re_75.append(sorted_x[int(len(sorted_x) * 0.75)])
                re_100.append(sorted_x[-1])

            return re_mean, re_0, re_25, re_50, re_75, re_100

        function_values = [result.heights for result in self.results]
        distance_values = [result.distances for result in self.results]

        function_values = pad_list(function_values)

        length = len(function_values[0])
        x_values = np.arange(downsampling - 1, length *
                             downsampling, downsampling)

        re_mean, re_0, re_25, re_50, re_75, re_100 = calc(
            function_values, 'max')

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
        fig.suptitle(f'Performance of {self.algorithm.name}')

        ax1.set_xscale('log')  # log scale for x axis
        ax1.plot(x_values, re_mean, label='mean')
        ax1.plot(x_values, re_0, label='0%')
        ax1.plot(x_values, re_25, label='25%')
        ax1.plot(x_values, re_50, label='50%')
        ax1.plot(x_values, re_75, label='75%')
        ax1.plot(x_values, re_100, label='100%')
        ax1.axhline(y=1344.9, color='r', linestyle='--', label='Ben Nevis')
        ax1.axhline(y=1309, color='#FFA500',
                    linestyle='--', label='Ben Macdui')
        ax1.fill_between(x_values, re_25, re_75, color='#5CA4FA', alpha=0.5)
        ax1.fill_between(x_values, re_0, re_100, color='#5CF7FA', alpha=0.25)
        ax1.legend(loc='lower right')
        ax1.set_xlabel('Number of evaluations')
        ax1.set_ylabel('Height')

        distance_values = pad_list(distance_values)

        re_mean, re_0, re_25, re_50, re_75, re_100 = calc(
            distance_values, 'min')

        ax2.set_xscale('log')  # log scale for x axis
        ax2.set_yscale('log')  # log scale for y axis
        ax2.plot(x_values, re_mean, label='mean')
        ax2.plot(x_values, re_0, label='100%')
        ax2.plot(x_values, re_25, label='75%')
        ax2.plot(x_values, re_50, label='50%')
        ax2.plot(x_values, re_75, label='25%')
        ax2.plot(x_values, re_100, label='0%')
        ax2.fill_between(x_values, re_25, re_75, color='#5CA4FA', alpha=0.5)
        ax2.fill_between(x_values, re_0, re_100, color='#5CF7FA', alpha=0.25)
        ax2.legend(loc='upper right')
        ax2.set_xlabel('Number of evaluations')
        ax2.set_ylabel('Distance to Ben Nevis')
        ax2.set_ylim(10, 2e6)

        plt.show()

    def plot_stacked_graph(self):
        """Plot a stacked graph for all instances."""

        self.fetch_full_results()

        function_values = pad_list([result.heights for result in self.results])

        height_bins = [1000, 1100, 1150, 1215, 1235, 1297, 1310, 1340, 1350]
        height_labels = [
            '',
            '',
            '~ top 50 Munros',
            '~ top 25 Monros',
            'Ben Nevis Massif (top 9 Munros)',
            'Caringorm Plateau (top 6 Munros)',
            'Ben Macdui (2nd highest Munro)',
            'Very close to Ben Nevis',
            'Ben Nevis (highest Munro)',
        ]

        def get_cat(h):
            for i, x in enumerate(height_bins):
                if h <= x:
                    return i

        random_results = []
        for values in function_values:
            prefix = np.maximum.accumulate(values)
            random_results.append(prefix)

        temp = np.array(random_results).T

        group_cnts = []

        for x in temp:
            group_cnt = [0] * len(height_bins)
            for h in x:
                group_cnt[get_cat(h)] += 1

            for i in range(len(group_cnt) - 1):
                group_cnt[i + 1] += group_cnt[i]

            group_cnts.append(group_cnt)

        group_cnts = np.array(group_cnts).T

        legend_elements = []

        def index_to_color(i):
            return plt.cm.tab20c(1 - (i + 1) / (len(height_bins)))

        fig, ax = plt.subplots(1, 1)

        for i, height in enumerate(height_bins):
            cnts = group_cnts[i]

            ax.fill_between(
                range(len(cnts)),
                group_cnts[i - 1] if i > 0 else [0] * len(cnts),
                group_cnts[i],
                color=index_to_color(i)
            )

            low = height_bins[i - 1] if i > 0 else 0
            high = height

            legend_elements.append(
                Patch(
                    color=index_to_color(i),
                    label=f'{low}m - {high}m {height_labels[i]}'
                )
            )

        fig.legend(
            handles=legend_elements[::-1],
            loc=7,
        )

        fig.suptitle(
            'Number of runs reaching a certain height at each function '
            'evaluation for {} runs of {}'.format(
                len(self.results),
                self.algorithm.name
            ))
        fig.tight_layout()
        fig.subplots_adjust(right=0.75)
        plt.show()
