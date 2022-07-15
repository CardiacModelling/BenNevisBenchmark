import nevis
import numpy as np
import matplotlib.pyplot as plt

nevis.download_os_terrain_50()

f = nevis.linear_interpolant()
ben_x, ben_y = nevis.ben().grid
def dist_to_ben(x, y):
    return np.linalg.norm(np.array([x, y]) - np.array([ben_x, ben_y]))
x_max, y_max = nevis.dimensions()
np.random.seed(1)

def plot_random_method(points_list, function_values=None):
    """
    Plot random method results.

    Parameters
    ----------
    points_list : list of list of tuple of shape (m, n, 2)
        List of visited points to be plotted. m is the number of executions,
        and n is the nubmer of evaluations per execution.
    function_values : list of float
        List of function values to be plotted.
    
    Returns
    -------
    fig, (ax1, ax2) : tuple of matplotlib.pyplot.Figure and (ax1, ax2)
    """

    def calc(values_list, method):
        random_results = []
        for values in values_list:
            prefix = (np.maximum if method == 'max' else np.minimum).accumulate(values)
            random_results.append(prefix)
        
        temp = np.array(random_results).T

        re_mean = []
        re_0 = []
        re_25 = []
        re_75 = []
        re_100 = []
        for x in temp:
            re_mean.append(np.mean(x))
            sorted_x = np.sort(x)
            re_0.append(sorted_x[0])
            re_25.append(sorted_x[int(len(sorted_x) * 0.25)])
            re_75.append(sorted_x[int(len(sorted_x) * 0.75)])
            re_100.append(sorted_x[-1])

        return re_mean, re_0, re_25, re_75, re_100
    
    # Function values
    if function_values is None:
        print('Calculating function values...')
        function_values = []
        for points in points_list:
            function_values.append([f(x, y) for x, y in points])
    re_mean, re_0, re_25, re_75, re_100 = calc(function_values, 'max')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 10))
    fig.suptitle('Performance of grid search')

    ax1.set_xscale('log') # log scale for x axis
    ax1.plot(re_mean, label='mean')
    ax1.plot(re_0, label='0%')
    ax1.plot(re_25, label='25%')
    ax1.plot(re_75, label='75%')
    ax1.plot(re_100, label='100%')
    ax1.axhline(y=1344.9, color='r', linestyle='--', label='Ben Nevis')
    ax1.fill_between(range(len(re_mean)), re_25, re_75, color='#5CA4FA', alpha=0.5)
    ax1.fill_between(range(len(re_mean)), re_0, re_100, color='#5CF7FA', alpha=0.25)
    ax1.legend(loc='lower right')
    ax1.set_xlabel('Number of evaluations')
    ax1.set_ylabel('Height')
    

    # Distances to Ben Nevis
    print('Calculating distances to Ben Nevis...')
    distance_values = []
    for points in points_list:
        distance_values.append([dist_to_ben(x, y) for x, y in points])
    
    re_mean, re_0, re_25, re_75, re_100 = calc(distance_values, 'min')

    ax2.set_xscale('log') # log scale for x axis
    ax2.set_yscale('log') # log scale for y axis
    ax2.plot(re_mean, label='mean')
    ax2.plot(re_0, label='100%')
    ax2.plot(re_25, label='75%')
    ax2.plot(re_75, label='25%')
    ax2.plot(re_100, label='0%')
    ax2.fill_between(range(len(re_mean)), re_25, re_75, color='#5CA4FA', alpha=0.5)
    ax2.fill_between(range(len(re_mean)), re_0, re_100, color='#5CF7FA', alpha=0.25)
    ax2.legend(loc='upper right')
    ax2.set_xlabel('Number of evaluations')
    ax2.set_ylabel('Distance to Ben Nevis')
    ax2.set_ylim(10, 2e6)
    

    return fig, (ax1, ax2)
    
if __name__ == '__main__':
    def random_search(n):    
        return [(np.random.uniform(0, x_max), np.random.uniform(0, y_max)) for _ in range(n)]
    points = [random_search(int(1e4)) for _ in range(100)]

    t = nevis.Timer()
    plot_random_method(points)
    print(t.format())
    plt.show()