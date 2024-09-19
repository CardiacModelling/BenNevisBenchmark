# A `nevis`-based benchmark framework for optimisers

## Reproducing the figures and tables in the paper

Tested on Ubuntu 22.04.3 LTS with Python 3.10.12.

1. Use a Python virtual environment: 
`python3 -m venv venv && source venv/bin/activate`.
2. Install the required packages `pip install -r requirements.txt`. Alternatively, you may run `pip install numpy pandas matplotlib seaborn scipy nevis optuna optuna-dashboard pints nlopt jupyter ipython pymongo pybind11 python-dotenv`.
3. Download the required data for the `nevis` module by running `python3 -c "import nevis; nevis.download_os_terrain_50()"`, if you haven't done so.
4. Run the the algorithm of finding local optima and their basin of attraction (as specified in `basin-problem/README.md`). In other words, you need to run:
```bash
cd basin-problem
sudo apt install python3.10-dev g++
python setup.py install
```
and then run `python calculate.py` twice. Then `cd ..`.

5. Go to `./basin-problem/ipynb/` and run `fig-1.ipynb` and `table-1.ipynb` to produce Figure 1 and Table 1. 
6. Create a directory `./result/` by `mkdir result`.
5. Navigate to the `src` directory by `cd src`, then run `python3 run.py cmaes de da mlsl nm pso`. This script will run the hyper-parameter tuning process and save the run results for the best instance for each of the following algorithms: CMA-ES, Differential Evolution, Dual Annealing, MLSL, Nelder-Mead, and PSO (Nelder-Mead does not have hyper-parameters so the tuning process is skipped). To speed up the process, you can run the tuning process for different algorithms concurrently in separate terminals; for example, you could run `python3 run.py cmaes nm da` in one terminal and `python3 run.py pso mlsl de` in another. For a full list of available options, use `python3 run.py --help`. 
6. Go to `./ipynb/plots/` and run `all.ipynb`. You should then find in `./ipynb/plots/imgs/` three figures: `combined-agg.png`, `combined-hb.png` and `output-ert.png`, which correspond to Figures 2, 3, 4 in the paper. Tables 5 and 6 are also produced in this notebook.

## File structures
### `./ipynb`

Some Jupyter notebooks.

- `interval_size.ipynb` The number of grid points that fall in each height interval.
- `lowest_point.ipynb` The lowest point within `n` meters away from the peaks. Could be useful in selecting height threshold for successful runs.

#### `./ipynb/plots`

- `all.ipynb` Plot most tables and figures in the paper.
- `de.ipynb` Generating plots and animations for Differential Evolution.
- `nelder-mead-multi.ipynb` Generating plots and animations for Nelder-Mead.

#### `./ipynb/legacy`

This directory has some notebooks from early trials. Some algorithms used have been re-implemented in the framework.

- `baseline.ipynb` Some trials of baseline algorithms on this problem. 
- `baseline_grid.ipynb` Grid search as a baseline algorithm. The plot methods have been reimplemented in the framework, and the algorithm itself is too inefficient.
- `cmaes.ipynb` A run of CMA-ES with plots.
- `DIRECT.ipynb` DIRECT algorithms (with different variations implemented in `nlopt`).
- `fitting-with-shgo.ipynb` SHGO algorithm.
- `plot.ipynb` Results and plots from earlier experiments.
- `result.ipynb` Results and plots from earlier experiments.
- `simulated_annealing.ipynb` Simulated annealing and dual annealing.
- `tol-test.ipynb` Testing the effect of absolute/relative tolerance on local optimisers.
- `variable-boundary.ipynb` An attempt to generate multiple problem instances by setting random boundaries. Aborted.

### `./basin-problem`

Scripts for finding all the local optima and their basin of attraction on the grid.


### `./src`

Class definitions for the benchmarking framework. 

`./src/framework` contains class definitions for the framework and `./src/algorithms` contains algorithms defined using the framework. 

`./src/tutorial.ipynb` is a tutorial for using this framework. 
  
### `./img`

Images used in documents and Jupyter notebooks.

  
### `./doc`

Legacy documents and presentation slides about this project.
