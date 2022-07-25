import nevis
from config import GRID_SIDE
from run_grid_search import run_grid_search
from plot_random_method import plot_random_method, read_results
import matplotlib.pyplot as plt

run_grid_search()
points_list, function_values, distance_values = read_results(f'grid_search_{GRID_SIDE}_')
t = nevis.Timer()
plot_random_method(points_list, function_values, distance_values)
print(t.format())
plt.show()