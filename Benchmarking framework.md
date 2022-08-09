# Benchmarking Framework 

## Problems

- The main problem: Finding Ben Nevis on the whole GB height map.
- Local problems: Finding the highest point in a hand-picked region.
  - `nevis. schiehallion`
  - `nevis.macdui`
- (TODO) Tile problems: Finding the highest point in two-letter tiles, so that we have a large problem space. (low priority?)

## Algorithms

### List of Algorithms

- Grid search
- Random search
- Bayesian Optimization

- CMA-ES
- DIRECT
- SHGO (used with local optimizers)
- Simulated Annealing (can be used with local optimizers)
- Restart strategies
- (TODO) "Differential evolutionn
- (TODO) TGO "topographical global optimization"


- Local optimizers
  - BFGS
  - Nelder-Mead

### Hyper-Parameter Tuning

- Manual Search
  - needs understanding of the algorithms
  - hard to reproduce
- Grid Search
  - trying all possible combinations
  - computationally costly for higher dimensions
- Random Search
  - simple idea: independent draws from uniform density from parameter space
  - proved to have equal or better performance for manual or grid search (because often only a small number of hyper-parameters affect the performance)
  - random search is unreliable for training some complex models
- Bayesian Optimization
  - uses Gaussian processes and an acquisition function to derive the maximum of the objective function. Updates itself when a new observation is made
  - common acquisition functions:
    - probability of improvement (PI) (tends to be trapped in local optima)
    - expected improvement (EI) (preferred)
    - GP upper confidence bound (GP-UCB) (needs further hyper-parameters)

  - implementation
    - https://github.com/fmfn/BayesianOptimization

  - more complex than random search but likely to perform better


## Performance

- termination criteria: terminates the algorithms when a maximum function evaluation number is reached OR we have reached >= a certain height (maybe 1340, sufficiently close to Ben Nevis). The algorithm may also terminate by itself, which we should try to prevent (but might not be able to)
  - successful runs: we reach a certain height  (maybe 1317 for Ben Nevis and 1307 for Ben Nevis + Ben Macdui, see table in Appendix) before the algorithm terminates
  - unsuccessful runs: the algorithm terminates because the maximum function evaluation number is reached or by itself before reaching the designated height

- when we actually run the algorithms we simply record all the function evaluations in an array and slice it to the point when the termination criteria are reached (as if it terminates even when it did not) and then we classify the run as successful or unsuccessful 

### Metrics

- Maximum height reached at a certain number of function evaluations
  - data are aggregated (i.e. 0, 25, 50, 75, 100 percentiles and mean and std) across multiple runs
  - variant: the percentage of runs that are reach certain heights (e.g. 1000, 1100, 1200, 1300) at a a certain number of function evaluations
- Successful rate
  - (# of successful runs according to the table above) / total runs

- Success performance
  - mean (FEs for successful runs)*(# of total runs) / (# of successful runs)

- the average runtime of successful runs


### Visualization

- Convergence plots (for a single algorithm and multiple algorithms)
  - showing maximum height reached at a certain number of function evaluations (as above)

- Boxplots and histograms
  - showing the distribution of performance metrics across different runs 

- Plots for showing a single run
  - 2-D scatter plot and trajectory plot
  - 3-D plot using Google Earth

## Appendix

Height around Ben Nevis:

|      | radius | minimum height |
| ---: | -----: | -------------: |
|    0 |      0 |    1344.951376 |
|    1 |     10 |    1336.209441 |
|    2 |     25 |    1317.590389 |
|    3 |     50 |    1272.713262 |
|    4 |    100 |    1195.117503 |

Height around Ben Macdui:

|      | radius | minimum height |
| ---: | -----: | -------------: |
|    0 |      0 |    1309.008933 |
|    1 |     10 |    1308.352555 |
|    2 |     25 |    1307.353242 |
|    3 |     50 |    1305.608210 |
|    4 |    100 |    1297.653107 |

Height around the 3rd hill:

|      | radius | minimum height |
| ---: | -----: | -------------: |
|    0 |      0 |    1293.900024 |
|    1 |     10 |    1287.766844 |
|    2 |     25 |    1276.278256 |
|    3 |     50 |    1251.159347 |
|    4 |    100 |    1182.576007 |
