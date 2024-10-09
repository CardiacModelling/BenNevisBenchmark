import pprint
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from .config import MAX_FES, MAX_INSTANCE_FES, logger
from .randomiser import Randomiser
from tqdm import tqdm
import optuna
from collections import namedtuple
from io import StringIO


def pad_list(ls, mode='last'):
    """Making each list of the input list of lists have the same length, by
    padding with the last element of each list."""
    if not ls:
        return []
    length = max(len(lst) for lst in ls)
    if mode == 'last':
        return [
            np.append(lst, [lst[-1]] * (length - len(lst)))
            for lst in ls
        ]
    elif mode == 'terminate':
        return [
            np.append(lst, [1500] * (length - len(lst)))
            for lst in ls
        ]
    elif mode == 'judge':
        return [
            np.append(lst, [lst[-1] if 1340 <= lst[-1] <
                      1350 else 1500] * (length - len(lst)))
            for lst in ls
        ]
    elif mode == 'edge':
        # Ben Nevis should be an edge
        result = []
        for lst in ls:
            last_value = lst[-1]
            if 1340 <= last_value < 1350:
                padding_num = min(1000, length - len(lst))
                new_lst = np.append(np.append(lst, [last_value] * padding_num),
                                    [1500] * (length - len(lst) - padding_num))
            else:
                new_lst = np.append(lst, [1500] * (length - len(lst)))
            result.append(new_lst)
        return result
    else:
        raise ValueError('Unknown mode.')


def float_to_latex(float_number):
    # convert the float number to latex format
    import math
    if float('inf') == float_number:
        return "\\( +\\infty \\)"
    # Extracting the exponent part
    exponent = int(math.floor(math.log10(abs(float_number))))

    # Extracting the mantissa part
    mantissa = float_number / (10 ** exponent)

    # Formatting the float number in the LaTeX style format
    latex_formatted_number = "\\({:.2f} \\times 10^{{{}}}\\)".format(mantissa, exponent)
    return latex_formatted_number


class AlgorithmInstance:
    """
    Class for an algorithm instance. This class is used to represent an
    instance of an algorithm with a specific set of hyper-parameters, and
    is used to run the instance and generate a list of ``Result``s.

    Attributes
    ----------
    algorithm : Algorithm
        The algorithm this instance belongs to.
    trial : optuna.Trial
        The optuna ``Trial`` object, used to specify the hyper-parameters of
        this instance.
    instance_index : int
        The index of this instance. This is used to distinguish between
        different instances of the same algorithm. If None, the index will
        be the same as the trial id. This is useful when the trial is
        an optuna.FixedTrial object that you can specify hyper-parameters
        manually, which does not have a trial id. In this case you specify
        the index manually. Try to use a negative index to avoid conflicts
        with the trial id.
    cache_enabled : bool
        Whether to cache `restart_results' for this instance. Default is False.
        Only set to True when no more runs will be carried out for this instance.

    Methods
    -------
    __call__(result_index)
        Run this instance and return the result of the (result_index)-th run.

    run_next()
        Run this instance with the next result index and return the result.

    run(save_handler, max_instance_fes, restart, save_partial, does_prune, measure)
        Run this instance and save all results to self.results.

    make_results_partial()
        Make all results of this instance partial.

    load_results(save_handler, partial)
        Load all results saved for this instance.

    performance_measures(max_instance_fes)
        Return all the performance measures of the instance based on the results.

    plot_convergence_graph(downsampling, img_path)
        Plot a convergence graph across all results of the instance.

    plot_stacked_graph(img_path, mode)
        Plot a stacked graph for all results of the instance.

    print_results()
        Print all results of this instance.
    """

    def __init__(self, algorithm, trial: optuna.Trial, instance_index=None, cache_enabled=False):
        self.algorithm = algorithm

        self.results = []
        self.results_patial = False  # whether the results are partial

        self.trial = trial
        self.instance_index = instance_index

        self.cache_enabled = cache_enabled
        self._restart_results = []

    @property
    def info(self):
        try:
            return {
                'algorithm_name': self.algorithm.name,
                'algorithm_version': self.algorithm.version,
                'instance_index': self.trial._trial_id,
            }
        except AttributeError:
            return {'algorithm_name': self.algorithm.name,
                    'algorithm_version': self.algorithm.version,
                    'instance_index': self.instance_index,
                    }

    def __call__(self, result_index):
        """
        Run this instance and return the result.

        Parameters
        ----------
        result_index : int
            The index of the result. This is used to distinguish between
            different results of the same instance. random_seed and init_guess
            will be generated based on this index.

        Returns
        -------
        Result
            The result of this run.
        """

        # generate random seed and initial guess based on result index
        rand_seed = Randomiser.get_rand_seed(result_index)
        init_guess = Randomiser.get_init_guess(result_index)
        result = self.algorithm.func(rand_seed, init_guess, self.trial)
        result.set_info({**self.info, 'result_index': result_index})
        return result

    def run_next(self):
        """
        Run this instance with the next result index and return the result.
        """
        result_index = len(self.results)
        logger.debug(f'Running #{result_index}...')
        result = self(result_index)
        # logger.debug(f'Eval num: {result.eval_num}, height: {result.ret_height}')
        logger.debug(pprint.pformat(result.to_dict()))
        self.results.append(result)
        return result

    def run(
        self,
        save_handler=None,
        max_instance_fes=MAX_INSTANCE_FES,
        restart=False,
        save_partial=True,
        does_prune=True,
        measure='gary_ert',
    ):
        """
        Run this instance and save all results to self.results.

        Parameters
        ----------
        save_handler : SaveHandler
            The save handler to use for saving the results.
        max_instance_fes : int
            The maximum number of function evaluations for a single instance.
        restart : bool
            Whether to restart the instance, i.e. clear self.results before running.
        save_partial : bool
            Whether to save results partially.
        does_prune : bool
            Whether to prune the trial. Set to True if you want to use optuna's
            pruning feature in the hyper-parameter tuning. Not useful when the trial
            is a FixedTrial or FrozenTrial object.
        measure : string
            The performance measure to use for the hyper-parameter tuning. Only used for
            pruning.
        """

        if restart:
            self.results = []
            self.results_patial = False

        current_instance_fes = 0
        for result in self.results:
            current_instance_fes += result.eval_num

        logger.info(pprint.pformat(self.info))
        step = 0

        with tqdm(total=max_instance_fes) as pbar:
            pbar.update(current_instance_fes)
            while current_instance_fes < max_instance_fes:
                logger.debug(f'Current instance fes: {current_instance_fes}')
                result = self.run_next()
                current_instance_fes += result.eval_num
                pbar.update(result.eval_num)
                if save_handler is not None:
                    save_handler.save_result(result, partial=save_partial)
                if does_prune:
                    self.trial.report(
                        self.performance_measures()[measure], step)
                    if self.trial.should_prune():
                        raise optuna.TrialPruned()
                    step += 1

    def make_results_partial(self):
        """Make all results of this instance partial."""
        for result in self.results:
            result.points = np.array([])
            result.heights = np.array([])
            result.trajectory = np.array([])
        self.results_patial = True

    def load_results(self, save_handler, partial=True):
        """Load all results saved for this instance.

        Parameters
        ----------
        save_handler : SaveHandler
            The save handler to use for loading the results.
        partial : bool
            Whether to load results partially.
        """
        self.results = save_handler.find_results(self.info, partial=partial)
        if self.results:
            self.results_patial = partial

    @property
    def restart_results(self):
        """For an early-terminating algorithm, turn all the runs into the runs of the
        multi-start version of the algorithm."""
        if self.cache_enabled and self._restart_results != []:
            return self._restart_results
        RestartResult = namedtuple('RestartResult', [
            'gary_score',
            'is_success',
            'eval_num',
            'ret_height',
            'max_height',
            'heights',
            'distances',
            'end_of_iterations',
            'points',
        ])
        restart_results = []

        gary_score, is_success, eval_num, ret_height = 0, False, 0, 0
        max_height = 0
        heights, distances = np.array([]), np.array([])
        end_of_iterations = []
        points = None

        for result in self.results:
            gary_score = max(gary_score, result.gary_score)
            ret_height = max(ret_height, result.ret_height)
            max_height = max(max_height, result.max_height)
            is_success = (is_success or result.is_success)
            heights = np.concatenate((heights, result.heights))
            if points is None:
                points = np.array(result.points)
            else:
                points = np.concatenate((points, result.points))
            distances = np.concatenate((distances, result.distances))
            end_of_iterations.extend(list(j + eval_num for j in result.end_of_iterations))
            eval_num += result.eval_num

            # We will discard the last `restart run' if it is not long enough
            # to either be successful or reach MAX_FES
            if is_success or eval_num >= MAX_FES:
                restart_results.append(RestartResult(
                    gary_score=gary_score,
                    is_success=is_success,
                    eval_num=eval_num,
                    ret_height=ret_height,
                    heights=heights,
                    distances=distances,
                    points=points,
                    end_of_iterations=end_of_iterations,
                    max_height=max_height,
                ))
                gary_score, is_success, eval_num, ret_height = 0, False, 0, 0
                max_height = 0
                heights, distances = np.array([]), np.array([])
                end_of_iterations = []
                points = None
        self._restart_results = restart_results
        return restart_results

    def performance_measures(self, max_instance_fes=None, using_restart_results=False):
        """
        Return all the performance measures of the instance. It's safe to run this
        method if even the results are ``partial''.

        Parameters
        ----------
        max_instance_fes : int
            The maximum number of function evaluations for a single instance. If
            not None, the results will be truncated to have at most this number
            of function evaluations.

        Returns
        -------
        dict
            A dictionary of performance measures. Keys are the names of the
            measures.
            The keys include:
            - 'success_rate': The success rate of the instance.
            - 'failure_rate': The failure rate of the instance.
            - 'success_cnt': The number of successful runs.
            - 'avg_success_eval': The average number of function evaluations.
               for successful runs.
            - 'hv': The hypervolume.
            - 'par2': Penalized average runtime with a factor of 2.
            - 'par10': Penalized average runtime with a factor of 10.
            - 'avg_height': The average height of the returned points.
            - 'ert': The expected runtime.
            - 'sp': The success performance.
            - 'success_rate_upper': The upper bound of the success rate (95% CI, Wilson score interval).
            - 'success_rate_lower': The lower bound of the success rate (95% CI, Wilson score interval).
            - 'success_rate_length': The length of the success rate interval (95% CI, Wilson score interval).
            - 'gary_ert': The GERT value based on Gary's score.
        """

        raw_results = self.restart_results if using_restart_results else self.results
        results = []
        if max_instance_fes is not None:
            for result in raw_results:
                if max_instance_fes > 0:
                    results.append(result)
                    max_instance_fes -= result.eval_num
                else:
                    break
        else:
            results = raw_results

        run_num = len(results)

        success_evals = []
        failed_evals = []
        ret_heights = []
        max_heights = []

        gary_score_sum = 0

        for result in results:
            is_success, eval_cnt = result.is_success, result.eval_num
            if is_success:
                success_evals.append(eval_cnt)
            else:
                failed_evals.append(eval_cnt)
            ret_heights.append(result.ret_height)
            max_heights.append(result.max_height)

            gary_score_sum += result.gary_score

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
                'avg_height': np.mean(ret_heights),
                'avg_max_height': np.mean(max_heights),
                'ert': float('inf'),
                'sp': float('inf'),
                # 'ert_std': float('inf'),
                'success_rate_upper': 0,
                'success_rate_lower': 0,
                'success_rate_length': 0,

                'gary_ert': float('inf') if gary_score_sum == 0 else (
                    np.sum(success_evals) + np.sum(failed_evals)) / gary_score_sum,
            }

        # success_eval_var = np.var(success_evals, ddof=1)
        # failed_eval_var = np.var(failed_evals, ddof=1)
        # avg_failed_eval_sqr = np.mean(np.square(failed_evals))

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
            'avg_max_height': np.mean(max_heights),
            'ert': avg_success_eval + (
                (1 - success_rate) / success_rate * avg_failed_eval
            ),
            'sp': avg_success_eval / success_rate,
            # 'ert_std': np.sqrt(
            #     (1 - success_rate) / success_rate * failed_eval_var + (
            #         (1-success_rate) / (success_rate**2) * avg_failed_eval_sqr
            #     ) + success_eval_var),
            'success_rate_upper': center + radius,
            'success_rate_lower': center - radius,
            'success_rate_length': radius * 2,

            'gary_ert': (np.sum(success_evals) + np.sum(failed_evals)) / gary_score_sum,
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

    def plot_convergence_graph(self, downsampling=1, img_path=None, using_restart_results=False):
        """
        Plot a convergence graph across all instances.

        Parameters
        ----------
        downsampling : int
            Downsampling factor on number of function evaluations.
        """

        assert not self.results_patial, "Results must be fully loaded."

        results = self.restart_results if using_restart_results else self.results

        def calc(values_list, method):
            results_ = []
            for values in values_list:
                prefix = method.accumulate(values)
                results_.append(prefix)

            temp = np.array(results_).T

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

        function_values = [
            result.heights[:MAX_FES] for result in results]
        distance_values = [
            result.distances[:MAX_FES] for result in results]

        function_values = pad_list(function_values)

        length = len(function_values[0])
        x_values = np.arange(downsampling - 1, length *
                             downsampling, downsampling)

        re_mean, re_0, re_25, re_50, re_75, re_100 = calc(
            function_values, np.maximum)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
        fig.suptitle(f'Aggregated convergence graphs for {self.algorithm.name}')

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
        ax1.set_xlabel('Number of function evaluations')
        ax1.set_ylabel('Height')

        distance_values = pad_list(distance_values)

        re_mean, re_0, re_25, re_50, re_75, re_100 = calc(
            distance_values, np.minimum)

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
        ax2.set_xlabel('Number of function evaluations')
        ax2.set_ylabel('Distance to Ben Nevis')
        ax2.set_ylim(10, 2e6)

        plt.savefig(img_path, bbox_inches='tight') if img_path else plt.show()

    def plot_stacked_graph(self, img_path=None, mode='last',
                           using_restart_results=False, with_legends=True, fig=None, ax=None):
        """Plot a stacked graph for all instances."""

        assert not self.results_patial, "Results must be fully loaded."
        results = self.restart_results if using_restart_results else self.results
        # we first truncate the results and then do a cumulative maximum
        results_accumulated = [np.maximum.accumulate(result.heights[:MAX_FES])
                               for result in results]
        function_values = pad_list(results_accumulated, mode=mode)
        # $[0, 600)$ & Lowland areas\\
        # $[600, 1000)$ & Mountainous areas\\
        # $[1000, 1100)$ & Approximately top 135 Munros \& 5 Welsh `Furths' \\
        # $[1100, 1150)$ & Approximately top 50 Munros \\
        # $[1150, 1215)$ & Approximately top 25 Munros \\
        # $[1215, 1235)$ & Wider Ben Nevis Massif (top 9 Munros) \\ % 1 point
        # $[1235, 1297)$ & Caringorm Plateau (top 6 Munros) \\ % 2 points
        # $[1297, 1310)$ & Ben Macdui (2nd highest Munro) \\ % 3 points
        # $[1310, 1340)$ & On Ben Nevis but not quite at the summit \\ % 7 points
        # $[1340, 1345)$ & Ben Nevis (highest Munro) \\ % 10 points
        # each value reprents the upper bound of an interval
        height_bins = [
            600,
            1000,
            1100,
            1150,
            1215,
            1235,
            1297,
            1310,
            1340,
            1350,
            2000,
        ]
        height_labels = [
            'Lowland areas',
            'Mountainous areas',
            'Approximately top 135 Munros & 5 Welsh \'Furths\'',
            'Approximately top 50 Munros',
            'Approximately top 25 Munros',
            'Wider Ben Nevis Massif (top 9 Munros)',
            'Caringorm Plateau (top 6 Munros)',
            'Ben Macdui (2nd highest Munro)',
            'On Ben Nevis but not quite at the summit',
            'Ben Nevis (highest Munro)',
            '(Terminated)'
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

        if fig is None or ax is None:
            fig, ax = plt.subplots(1, 1)

        for i, height in enumerate(height_bins):
            cnts = group_cnts[i]

            low = height_bins[i - 1] if i > 0 else -100
            high = height

            if height_labels[i] != '(Terminated)':
                label = f'{low}m - {high}m\n{height_labels[i]}'
                color = index_to_color(i)
            else:
                label = '(Terminated)'
                color = (1, 1, 1, 1)

            ax.fill_between(
                range(len(cnts)),
                group_cnts[i - 1] if i > 0 else [0] * len(cnts),
                group_cnts[i],
                color=color,
            )

            legend_elements.append(
                Patch(
                    color=color,
                    label=label,
                )
            )

        if with_legends:
            fig.legend(
                handles=legend_elements[::-1],
                loc='upper left',
                bbox_to_anchor=(1, 0.9)
            )

        ax.set_title(
            'Height-band graph for {}'.format(
                self.algorithm.name
            ))

        ax.set_xlabel('Number of function evaluations')
        ax.set_ylabel('Number of runs')
        plt.savefig(img_path, bbox_inches='tight') if img_path else plt.show()

    def params_to_latex(self, int_fields=[]):
        d = self.trial.params
        output = StringIO()
        for k, v in d.items():
            kk = k.replace('_', '\\_')
            if k in int_fields:
                print(f'& \\texttt{{{kk}}} & {v} \\\\', file=output)
            else:
                print(f'& \\texttt{{{kk}}} & {float_to_latex(v)} \\\\', file=output)
        return output.getvalue()

    def performance_to_latex(self, using_restart_results=False):
        results = self.restart_results if using_restart_results else self.results
        run_num = len(results)
        performance_dict = self.performance_measures(using_restart_results=using_restart_results)
        sr = performance_dict['success_rate']
        ah = performance_dict['avg_max_height']
        ert = performance_dict['ert']
        gert = performance_dict['gary_ert']

        output = StringIO()

        def print_to_str(s, end):
            print(s, end=end, file=output)

        print_to_str(self.algorithm.name.replace('_', ' '), end='\t')
        print_to_str(f'& {run_num}', end='\t')
        print_to_str(f'& {round(sr * 100)}\\%', end='\t')
        print_to_str(f'& {round(ah)}', end='\t')
        print_to_str(f'& {float_to_latex(ert)} ', end='\t')
        print_to_str(f'& {float_to_latex(gert)} ', end='\\\\')
        return output.getvalue()
