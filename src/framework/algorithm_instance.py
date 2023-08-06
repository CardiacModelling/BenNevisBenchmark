from functools import cache
from pprint import pprint
import nevis
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from .config import MAX_FES, RUN_NUM
import time
from tqdm import tqdm


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
    def __init__(self, algorithm, params, save_handler, hash=None):
        """
        Class for an algorithm instance. All results of this instance
        previously saved will be loaded automatically.

        Parameters
        ----------
        algorithm : Algorithm
            The algorithm this instance belongs to.
        params : dict
            The hyper-paramters of this instance.
        save_handler : SaveHandler
            The save handler of the algorithm.
        hash : int
            The hash of this instance. If not provided, a hash will be
            generated based on the time.
        """
        self.algorithm = algorithm
        self.params: dict = params

        self.info = {
            'algorithm_name': algorithm.name,
            'algorithm_version': algorithm.version,
            **params,
        }
        if hash is None:
            self.hash = time.time()
        else:
            self.hash = hash

        self.save_handler = save_handler

        self.results_patial = False
        self.results = set()
        self.load_results()

    def __eq__(self, other) -> bool:
        return self.hash == other.hash

    def __hash__(self):
        return int(self.hash * 1000000)

    def run(self, run_num=RUN_NUM):
        """
        Run this instance and save all results. Notice that this function will
        not  run the instance if there are already enough results saved for
        this instance.

        Parameters
        ----------
        run_num : int
            The number of times to run this instance.
        """
        self.save_handler.save_instance(self)

        if len(self.results) >= run_num:
            return

        print(f'Running instance {self.hash}')
        pprint(self.info)
        remain_run_times = run_num - len(self.results)
        for _ in tqdm(range(remain_run_times)):
            result = self.algorithm(**self.params)
            self.results.add(result)
            self.save_handler.add_result(self, result)

    def make_results_partial(self):
        for result in self.results:
            result.turn_partial()

    def fetch_full_results(self):
        if self.results_patial:
            self.load_results(False)

    def load_results(self, partial=True):
        """Load all results saved for this instance."""
        results = self.save_handler.load_results(self.hash, partial)
        self.results = results
        if results:
            self.results_patial = partial

    @cache
    def performance_measures(
        self,
        run=True,
    ):
        """
        Return all the performance measures of the instance.

        Parameters
        ----------
        run : bool
            Whether to run the instance first if there are not enough results
            saved.

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
        if run:
            self.run()
        results = list(self.results)
        run_num = len(results)

        success_cnt = 0
        success_eval_cnt = 0
        failed_cnt = 0
        failed_eval_cnt = 0
        height_sum = 0
        for result in results:
            is_success, eval_cnt = result.is_success, result.eval_num
            if is_success:
                success_cnt += 1
                success_eval_cnt += eval_cnt
            else:
                failed_cnt += 1
                failed_eval_cnt += eval_cnt

            height_sum += result.ret_height

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
            }

        success_rate = success_cnt / run_num
        avg_success_eval = success_eval_cnt / success_cnt
        avg_failed_eval = failed_eval_cnt / failed_cnt if failed_cnt != 0 \
            else 0
        return {
            'success_rate': success_rate,
            'failure_rate': 1 - success_rate,
            'success_cnt': success_cnt,
            'avg_success_eval': avg_success_eval,
            'hv': (MAX_FES - avg_success_eval) * success_rate,
            'par2': (failed_cnt * 2 * MAX_FES + success_eval_cnt) / run_num,
            'par10':  (failed_cnt * 10 * MAX_FES + success_eval_cnt) / run_num,
            'avg_height': height_sum / run_num,
            'ert': avg_success_eval + (
                1 - success_rate) / success_rate * avg_failed_eval,
            'sp': avg_success_eval / success_rate,
        }

    def print(self):
        """
        Print the information of this instance.
        """
        pprint(self.hash)
        pprint(self.info)

    def print_results(self):
        """Print all results of this instance."""
        self.fetch_full_results()

        pprint(self.performance_measures())
        for i, result in enumerate(self.results):
            print(f'=== Result #{i} ===')
            print(f'{result.len_points}')
            result.print()
            print()

    def plot_histogram(self):
        """Plot the histogram of the heights, distances to Ben Nevis, and
        numbers of function of evaluations of all results."""
        heights = []
        distances = []
        evals = []

        for result in self.results:
            heights.append(result.ret_height)
            distances.append(result.ret_distance)
            evals.append(result.len_points)

        fig, axs = plt.subplots(1, 3, figsize=(18, 10))

        axs[0].hist(heights)
        axs[1].hist(distances)
        axs[2].hist(evals)
        fig.suptitle(
            f'Histogram of returned heights, distances to Ben Nevis, and'
            f'numbers of function evals for {len(self.results)} runs of '
            f'{self.algorithm.name}')
        plt.show()

    def plot_ret_points(self):
        """
        Plot a map of the returned points of all results.
        """
        ret_points = []

        for result in self.results:
            ret_points.append(result.ret_point)

        nevis.plot(
            labels={
                'Ben Neivs': nevis.ben(),
                'Ben Macdui': nevis.Hill.by_rank(2).coords,
            },
            points=np.array(ret_points)
        )

        plt.show()

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
