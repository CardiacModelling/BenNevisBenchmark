import nevis
import numpy as np
import math
import json
import os

nevis.download_os_terrain_50()
f = nevis.linear_interpolant()
x_max, y_max = nevis.dimensions()
np.random.seed(1)

# side length of the grid
SIDE = 5000
# times of shuffling
N = 80

def make_pairs(s):
    # s is the side length of the square
    xs = np.linspace(0, x_max, math.ceil(x_max / s))
    ys = np.linspace(0, y_max, math.ceil(y_max / s))

    # The Cartesian product of the two lists
    pairs = np.dstack(np.meshgrid(xs, ys)).reshape(-1, 2)
    return pairs


pairs = make_pairs(SIDE)
re_len = len(pairs)
print('Evaluating...')
re_worst = [f(x, y) for x, y in pairs]
re_worst.sort()
func_values = re_worst.copy()

def gen_res(re_worst, method='max'):
    random_results = []
    print('Shuffling...')
    for _ in range(N):
        shuffled_f = np.random.permutation(re_worst)
        prefix = (np.maximum if method == 'max' else np.minimum).accumulate(shuffled_f)
        random_results.append(prefix)

    temp = np.array(random_results).T

    re_mean = []
    re_0 = []
    re_25 = []
    re_75 = []
    re_100 = []
    print('Calculating...')
    for x in temp:
        re_mean.append(np.mean(x))
        sorted_x = np.sort(x)
        re_0.append(sorted_x[0])
        re_25.append(sorted_x[int(len(sorted_x) * 0.25)])
        re_75.append(sorted_x[int(len(sorted_x) * 0.75)])
        re_100.append(sorted_x[-1])

    # re_best = np.repeat((np.max if method == 'max' else np.min)(re_worst), re_len)
    re_best = [(np.max if method == 'max' else np.min)(re_worst)] * re_len
    print('best value obtained: ', re_best[0])
    return re_mean, re_0, re_25, re_75, re_100, re_best

result = {}

re_worst = func_values
re_mean, re_0, re_25, re_75, re_100, re_best = gen_res(re_worst)

result['height'] = {
    'mean': re_mean,
    '0': re_0,
    '25': re_25,
    '75': re_75,
    '100': re_100,
    'best': re_best,
    'worst': re_worst,
}

ben_x, ben_y = nevis.ben().grid
def dist_to_ben(x, y):
    return ((x - ben_x)**2 + (y - ben_y)**2) ** 0.5

distances = [dist_to_ben(x, y) for x, y in pairs]
distances.sort()
distances.reverse()

re_worst = distances
re_mean, re_0, re_25, re_75, re_100, re_best = gen_res(re_worst, 'min')

result['distance'] = {
    'mean': re_mean,
    '0': re_0,
    '25': re_25,
    '75': re_75,
    '100': re_100,
    'best': re_best,
    'worst': re_worst,
}

print('Saving...')
# make directory if it doesn't exist
if not os.path.exists('result'):
    os.makedirs('result')
json.dump(result, open('./result/baseline_grid_calc.json', 'w'))