import numpy as np
import nevis
import math
import time
import pickle
import os
from config import GRID_SIDE as SIDE, GRID_N as N

def run_grid_search():
    x_max, y_max = nevis.dimensions()

    m = SIDE // 50
    print('Getting one function result...')
    function_result = (nevis.gb()[::m, ::m])
    function_result = np.reshape(function_result, (np.prod(function_result.shape,)))

    def genrate_shuffled_results(one_result, method='max'):
        rng = np.random.default_rng()
        random_results = rng.permuted(np.tile(one_result, (N, 1)), axis=1)
        random_results = (np.maximum if method == 'max' else np.minimum).accumulate(random_results, axis=1)
        return random_results

    print("Getting shuffled function results...")
    function_shuffled_results = genrate_shuffled_results(function_result, method='max')

    def make_pairs(s):
        # s is the side length of the square
        xs = np.linspace(0, x_max, math.ceil(x_max / s))
        ys = np.linspace(0, y_max, math.ceil(y_max / s))

        # The Cartesian product of the two lists
        pairs = np.dstack(np.meshgrid(xs, ys)).reshape(-1, 2)
        return pairs

    ben_x, ben_y = nevis.ben().grid
    def dist_to_ben(x, y):
        return ((x - ben_x)**2 + (y - ben_y)**2) ** 0.5

    print("Getting one distance result...")
    pairs = make_pairs(SIDE)
    distance_result = np.array([dist_to_ben(x, y) for x, y in pairs])

    print("Getting shuffled distance results...")
    distance_shuffled_results = genrate_shuffled_results(distance_result, method='min')

    data = {
        "points_list": [],
        "function_values": function_shuffled_results,
        "distance_values": distance_shuffled_results,
    }


    print("Saving data...")
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    os.makedirs('../result', exist_ok=True)
    pickle.dump(data, open(f"../result/grid_search_{SIDE}_{N}_{timestamp}.pickle", "wb"))


if __name__ == '__main__':
    run_grid_search()