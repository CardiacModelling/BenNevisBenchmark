# Benchmark Framework for Comparing Optimization Algorithms on `nevis`

## Reproducing the figures and tables in the paper

An Ubuntu environment is assumed.

1. Use a Python virtual environment: 
`python3 -m venv venv && source venv/bin/activate`.
2. Install the required packages `pip install -r requirements.txt`.
3. Run the the algorithm of finding local optima and their basin of attraction (as specified in `basin-problem/README.md`). In other words, you need to run:
```bash
cd basin-problem
sudo apt install python3.10-dev g++
python setup.py install
makedir res
```
and then run `python calculate.py` twice. Then `cd ..`.

4. Go to `./basin-problem/ipynb/` and run `fig-1.ipynb` and `table-1.ipynb` to produce Figure 1 and Table 1. 
4. Create a directory `./result/`.
5. `cd src`, then run `python3 run.py`. This script will run the hyper-parameter tuning process for each of the following algorithms: CMA-ES, Differential Evolution, Dual Annealing, MLSL, Nelder-Mead, and PSO.
6. Go to `./ipynb/plots/` and run `all.ipynb`. You should then find in `./ipynb/plots/imgs/` three figures: `combined-agg.png`, `combined-hb.png` and `output-ert.png`, which correspond to Figures 2, 3, 4 in the paper. Tables 5 and 6 are also produced in this file.

## `./doc`

Documents and presentation slides about this project.

## `./ipynb`

Some Jupyter notebooks.

- `baseline.ipynb` Some trials of baseline algorithms on this problem. One of them has been reimplemented in `./src/algorithms/multistart.py`, and other methods are too inefficient to further investigate.
- `baseline_grid.ipynb` Grid search as a baseline algorithm. The plot methods have been reimplemented in the framework, and the algorithm itself is too inefficient.
- `DIRECT.ipynb` DIRECT algorithms (with different variations implemented in `nlopt`).
- `fitting-with-shgo.ipynb` SHGO algorithm (see `./src/algorithms/shgo.py`).
- `lowest_point.ipynb` The lowest point within `n` meters away from the peaks. Could be useful in selecting height threshold for successful runs.
- `simulated_annealing.ipynb` Simulated annealing and dual annealing (see `./src/algorithms/simulated_annealing.py` and `./src/algorithms/dual_annealing.py`).

## `./img`

Images used in documents and Jupyter notebooks.

## `./src`

Class definitions for the benchmarking framework.


- **An algorithm** contains
  - a function that takes hyper-parameters and returns an optimization result
  - a hyper-parameter space (the range of values they can take)
  - a name and a version number
  - a save handler, which creates a directory for this algorithm and saves and loads all the instances and results
  - multiple **algorithm instances** (or one if there is no hyper-parameter)

- **An algorithm instance** contains

  - an algorithm
  - a particular set of hyper-parameter
  - a hash (either specified by user or generated using the current time stamp)
  - the save handler of its algorithm
  - multiple **run results** (or one if it is a deterministic algorithm)

- **A run result** contains
  - the found optimization point and value
  - a list of all visited points
  - a message that explains why the run was terminated
  - the heights and distances to Ben Nevis of all visited points
  - the time stamp when it is run, used as an identifier

  ---

- **A run result** can

  - be classified as successful or failed (see `Result.succeess_eval`)
  - be visualized using 2D plots or Google Earth (see `Result.plot_global`, `Result.plot_partial`, and `Result.generate_kml`)

- **An algorithm instance** can

  - run and obtain multiple results (see `AlgorithmInstance.run`)
  - calculate its performance measure using (the classification of) its run results (see `AlgorithmInstance.performance_measures`)
  - plot its convergence graph and stacked graph using its run results (See `AlgorithmInstance.plot_convergence_graph` and `AlgorithmInstance.plot_stacked_graph`)
  - plot the histogram of the heights, distances to Ben Nevis, and numbers of function of evaluations of all results (see `AlgorithmInstance.plot_histogram`)
  - plot a map of the returned points of all results (see `AlgorithmInstance.plot_ret_points`)

- **An algorithm** can


  - generate an instance with user specified hyper-parameters (see `Algorithm.generate_instance`), or generate a random instance within the hyper-parameter space (see `Algorithm.generate_random_instance`)
  - tune its hyper-parameters by generating random algorithm instances and picking the one with the best performance measure (see `Algorithm.tune_params`)
  - visualize its hyper-parameter tuning (up to 2 hyper-parameters), by showing a scatter plot with one hyper-parameter on each axis and the color or area of the marks representing designated performance measures (see `Algorithm.plot_tuning`)
  - make a scatter plot of two performance measures across its instances (so as to investigate the correlation between the two performance measures) (see `Algorithm.plot_two_measures`), or make a pair plot of all available performance measures (see `Algorithm.plot_all_measures`)

- The saving and loading of instances and results can be found in the class `SaveHandler`.

  ---

  To define a new algorithm, you can make use of the decorator  `optimizer` defined in `runner.py`. It will help you record visited points. Existing algorithms are defined in `./src/algorithms` which you can refer to as examples. Notice that the function wrapped by the decorator `optimizer` needs to be a minimizer instead of a maximizer.

  The constants are defined in `./src/framework/config.py`.

  Some examples of using this framework is shown in `./src/main.ipynb`.

  

  

  

  
