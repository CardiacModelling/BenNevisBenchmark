from functools import cache
from pprint import pprint
import numpy as np
import matplotlib.pyplot as plt
from .config import MAX_FES, SUCCESS_HEIGHT, RUN_NUM
import time
from tqdm import tqdm


class AlgorithmInstance:
    def __init__(self, algorithm, params, save_handler, hash=None):
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
        self.results = set()

        self.save_handler = save_handler

    def __eq__(self, other) -> bool:
        return self.hash == other.hash
    
    def __hash__(self):
        return int(self.hash * 1000000)

    def run(self, run_num):
        self.save_handler.save_instance(self)

        if len(self.results) >= run_num:
            return
        print(f'Running instance {self.hash}')
        pprint(self.info)
        remain_run_times = run_num - len(self.results)
        for _ in tqdm(range(remain_run_times)):
            result = self.algorithm(**self.params)
            # result.print()
            self.results.add(result)
            self.save_handler.add_result(self, result)
        
        
    @cache
    def success_measures(self,
        max_fes=MAX_FES,
        success_height=SUCCESS_HEIGHT,
        run_num=RUN_NUM
    ):
        self.run(run_num)
        results = list(self.results)[:run_num]
        
        success_cnt = 0
        success_eval_cnt = 0
        height_sum = 0
        for result in results:
            is_success, eval_cnt = result.success_eval(max_fes, success_height)
            if is_success:
                success_cnt += 1
                success_eval_cnt += eval_cnt
            height_sum += result.ret_heigt

        
        success_rate = success_cnt / run_num
        performance = float('inf') if success_cnt == 0\
            else success_eval_cnt / success_cnt * run_num / success_cnt
        if success_cnt == 0:
            avg_success_eval = max_fes
        else:
            avg_success_eval = success_eval_cnt / success_cnt
        return {
            'success_rate': success_rate,
            'failure_rate': 1 - success_rate,
            'success_cnt': success_cnt,
            'avg_success_eval': avg_success_eval,
            'hv': (max_fes - avg_success_eval) * success_rate,
            'performance': performance,
            'par2': ((run_num - success_cnt) * 2 * max_fes + success_eval_cnt) / run_num,
            'par10':  ((run_num - success_cnt) * 10 * max_fes + success_eval_cnt) / run_num,
            'avg_height': height_sum / run_num,
            'ert': avg_success_eval + (1 - success_rate) / success_rate * max_fes if success_cnt != 0 else float('inf')
        }

    def print(self):
        pprint(self.info)
        for i, result in enumerate(self.results):
            print(f'=== Result #{i} ===')
            print(f'{len(result.points)}')
            result.print()
            print()


    def plot_convergence_graph(self, downsampling=1):
        """
        Plot random method results.

        Parameters
        ----------
        downsampling : int
            Downsampling factor.
        
        Returns
        -------
        fig, (ax1, ax2) : tuple of matplotlib.pyplot.Figure and (ax1, ax2)
        """
        def pad_list(ls):
            if not ls:
                return []
            length = max(len(l) for l in ls)
            return [
                np.append(l, [l[-1]] * (length - len(l)))
                for l in ls
            ]

        def calc(values_list, method):
            random_results = []
            for values in values_list:
                prefix = (np.maximum if method == 'max' else np.minimum).accumulate(values)
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
        x_values = np.arange(downsampling - 1, length * downsampling, downsampling)
        
        print(f'Length of function_values: {len(function_values)}')
        re_mean, re_0, re_25, re_50, re_75, re_100 = calc(function_values, 'max')

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 10))
        fig.suptitle(f'Performance of {self.algorithm.name}')

        ax1.set_xscale('log') # log scale for x axis
        ax1.plot(x_values, re_mean, label='mean')
        ax1.plot(x_values, re_0, label='0%')
        ax1.plot(x_values, re_25, label='25%')
        ax1.plot(x_values, re_50, label='50%')
        ax1.plot(x_values, re_75, label='75%')
        ax1.plot(x_values, re_100, label='100%')
        ax1.axhline(y=1344.9, color='r', linestyle='--', label='Ben Nevis')
        ax1.axhline(y=1309, color='#FFA500', linestyle='--', label='Ben Macdui')
        ax1.fill_between(x_values, re_25, re_75, color='#5CA4FA', alpha=0.5)
        ax1.fill_between(x_values, re_0, re_100, color='#5CF7FA', alpha=0.25)
        ax1.legend(loc='lower right')
        ax1.set_xlabel('Number of evaluations')
        ax1.set_ylabel('Height')
        
        
        distance_values = pad_list(distance_values)
        print(f'Length of distance_values: {len(distance_values)}')
        re_mean, re_0, re_25, re_50, re_75, re_100 = calc(distance_values, 'min')

        ax2.set_xscale('log') # log scale for x axis
        ax2.set_yscale('log') # log scale for y axis
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
        

        return fig, (ax1, ax2)
