from scipy.optimize import dual_annealing, minimize
import nevis
import time
import os
import pickle
import numpy as np


def run_dual_annealing():
    f = nevis.linear_interpolant()
    points = []
    function_values = []
    def wrapper(u):
        x, y = u
        points.append((x, y))
        z = f(x, y)
        function_values.append(z)
        return -z

    x_max, y_max = nevis.dimensions()
    ret = dual_annealing(
        wrapper, 
        bounds=[(0, x_max), (0, y_max)],
        maxiter=1500,
        initial_temp=31214.21,
        restart_temp_ratio=2.7826e-5,
    )

    x, y = ret.x
    z = -ret.fun

    nevis.print_result(x, y, z)

    
    data = {
        "points_list": [points],
        "function_values": [function_values],
        "distance_values": [],
    }


    print("Saving data...")
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    os.makedirs('../result', exist_ok=True)
    pickle.dump(data, open(f"../result/simulated_annealing_9_{timestamp}.pickle", "wb"))


def run_simulated_annealing():
    f = nevis.linear_interpolant()
    points = []
    function_values = []
    def wrapper(u):
        x, y = u
        points.append((x, y))
        z = f(x, y)
        function_values.append(z)
        return -z

    x_max, y_max = nevis.dimensions()

    def simulated_annealing(func, maxiter, step_size, initial_temp, restart_temp_ratio):
        best = np.random.uniform((0, 0), (x_max, y_max))
        best_eval = func(best)

        curr, curr_eval = best, best_eval

        i = 0

        restart_cnt = 0
        for _ in range(maxiter):

            t = initial_temp / float(i + 1)
            i += 1

            if t < initial_temp * restart_temp_ratio:
                local_ret = minimize(
                    func, 
                    curr, 
                    bounds=[(0, x_max), (0, y_max)],
                    method='Nelder-Mead',
                    options={'fatol': 1e-4, 'xatol': 1e-4}
                )
                if local_ret.success and local_ret.fun < best_eval:
                    # print('local search success')
                    best, best_eval = local_ret.x, local_ret.fun
                
                restart_cnt += 1
                t = initial_temp
                i = 0
                curr = np.random.uniform((0, 0), (x_max, y_max))
                curr_eval = func(curr)
                candidate, candidate_eval = curr, curr_eval
            
            else:
                candidate = np.random.normal(curr, step_size)
                # cx, cy = curr
                # candidate = np.random.uniform((cx - step_size, cy - step_size), (cx + step_size, cy + step_size))
                if candidate[0] < 0: candidate[0] = 0
                if candidate[1] < 0: candidate[1] = 0
                if candidate[0] > x_max: candidate[0] = x_max
                if candidate[1] > y_max: candidate[1] = y_max
                candidate_eval = func(candidate)
            
            if candidate_eval < best_eval:
                best, best_eval = candidate, candidate_eval
            
            diff = candidate_eval - curr_eval
            if diff < 0 or np.random.random() < np.exp(-diff / t):
                curr, curr_eval = candidate, candidate_eval
        
        print('restart count: ', restart_cnt)
        return best, best_eval

    (x, y), z = simulated_annealing(wrapper, 
        maxiter=20000, 
        step_size=2e4, 
        initial_temp=5e4, 
        restart_temp_ratio=1e-3
    )
    nevis.print_result(x, y, -z)

    data = {
        "points_list": [points],
        "function_values": [function_values],
        "distance_values": [],
    }


    print("Saving data...")
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    os.makedirs('../result', exist_ok=True)
    pickle.dump(data, open(f"../result/simulated_annealing_8_{timestamp}.pickle", "wb"))


if __name__ == '__main__':
    for i in range(100):
        run_dual_annealing()