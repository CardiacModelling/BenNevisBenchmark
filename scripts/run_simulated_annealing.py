from scipy.optimize import dual_annealing
import nevis
import time
import os
import pickle


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
    ret = dual_annealing(
        wrapper, 
        bounds=[(0, x_max), (0, y_max)],
        maxiter=2000,
        initial_temp=5e4,
        restart_temp_ratio=1e-4,
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
    pickle.dump(data, open(f"../result/simulated_annealing_6_{timestamp}.pickle", "wb"))


if __name__ == '__main__':
    for i in range(100):
        run_simulated_annealing()