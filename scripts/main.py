import nevis
from config import GRID_SIDE, GRID_DOWNSAMPLING
from run_grid_search import run_grid_search
from plot_random_method import plot_random_method, read_results
import matplotlib.pyplot as plt


run_grid_search()
points_list, function_values, distance_values, downsampling = read_results(f'grid_search_{GRID_SIDE}_{GRID_DOWNSAMPLING}_')
t = nevis.Timer()
plot_random_method(points_list, function_values, distance_values, method_name='Grid Search', downsampling=downsampling)
print(t.format())
plt.show()