import nevis
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from config import GRID_SIDE

nevis.download_os_terrain_50()

f = nevis.linear_interpolant()
ben_x, ben_y = nevis.ben().grid
def dist_to_ben(x, y):
    return np.linalg.norm(np.array([x, y]) - np.array([ben_x, ben_y]))
x_max, y_max = nevis.dimensions()
np.random.seed(1)


def pad_list(ls):
    if not ls:
        return []
    length = max(len(l) for l in ls)
    return [
        np.append(l, [l[-1]] * (length - len(l)))
        for l in ls
    ]

def plot_random_method(points_list=None, function_values=None, distance_values=None, method_name='',
    downsampling=1):
    """
    Plot random method results.

    Parameters
    ----------
    points_list : list of list of tuple of shape (m, n, 2)
        List of visited points to be plotted. m is the number of executions,
        and n is the nubmer of evaluations per execution. If None, ``function_values``
        and ``distance_values`` must be provided.
    function_values : list of list of float
        List of function values to be plotted. If None, they are obtained using
        ``points_list`` by evaluating the function at each point.
    distance_values : list of list of float
        List of distance values to be plotted. If None, they are obtained using
        ``points_list`` by calculating the distance to the Ben Nevis at each point.
    method_name : str
        Name of the method.
    downsampling : int
        Downsampling factor.
    
    Returns
    -------
    fig, (ax1, ax2) : tuple of matplotlib.pyplot.Figure and (ax1, ax2)
    """

    if points_list is None:
        assert function_values is not None and distance_values is not None, \
            "Either points_list or (function_values and distance_values) must be provided."

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
    
    # Function values
    if function_values is None:
        print('Calculating function values...')
        function_values = []
        for points in points_list:
            function_values.append([f(x, y) for x, y in points])
    
    function_values = pad_list(function_values)
    
    length = len(function_values[0])
    x_values = np.arange(downsampling - 1, length * downsampling, downsampling)
    
    print(f'Length of function_values: {len(function_values)}')
    re_mean, re_0, re_25, re_50, re_75, re_100 = calc(function_values, 'max')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 10))
    fig.suptitle(f'Performance of {method_name}')

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
    

    # Distances to Ben Nevis
    if distance_values is None:
        print('Calculating distances to Ben Nevis...')
        distance_values = []
        for points in points_list:
            distance_values.append([dist_to_ben(x, y) for x, y in points])
    
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


def read_results(prefix):
    """Read results from all pickle files with prefix ``prefix``."""   
    points_list = []
    function_values = []
    distance_values = []
    downsampling = None
    for file in os.listdir('../result/'):
        if file.startswith(prefix) and file.endswith('.pickle'):
            print('Reading {}...'.format(file))

            data  = pickle.load(open('../result/' + file, 'rb'))
            points_list.extend(data.get('points_list', []))
            function_values.extend(data.get('function_values', []))
            distance_values.extend(data.get('distance_values', []))
            if data.get('downsampling') is not None:
                if downsampling is None:
                    downsampling = data['downsampling']
                else:
                    assert downsampling == data['downsampling'], \
                        "All downsampling values must be the same."
        
    return points_list or None, function_values or None, distance_values or None, downsampling or 1
