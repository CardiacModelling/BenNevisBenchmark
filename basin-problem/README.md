Compile this library called `basin` (on Ubuntu):
```bash
cd basin-problem
sudo apt install python3.10-dev g++
pip install pybind11
python setup.py install
```
Then we can import it in Python. Basic usage:
```python
import basin
import nevis

h = nevis.gb()

# calculate the list of maxima and the steepest neighbour for each point
maxima, sn = basin.find_maxima(h)
# label each point with its b.o.a. 
# and calculate the sum of gradient ascending path lengths in each b.o.a.
label, path_sum = basin.find_basins(h, sn, maxima)
# calculate the area of each b.o.a. not excluding sea
area = basin.count_basin_area(label, len(maxima), data)
```

But these have been written in the script `calculate.py`! It will help you save the calculation to some `.npy` files. Note: you need to run 
```python
python calculate.py
``` 
twice to avoid memory overflow (at least I think this is why it would die if I try to calculate it all in one go on my machine).

There is also a testing script for a small example in `test.py`.

The note books in `./ipynb`:
- `check.ipynb` runs some checks to see if the calculated data is reasonable;
- `fig-1.ipynb` plots (entire map + Ben Nevis + Ben Macdui + height histogram) figure;
- `plot-boa.ipynb` gives a method for plotting the b.o.a. labels over the height map;
- `largest-area.ipynb` investigates the local maxima with the largest b.o.a. areas (excluding seas);
- `table-1.ipynb` calculates the data for Table 1 in the paper;
- `sea-maxima.ipynb` investigates the local maxima under sea level and 'flat' & 'strict' local maxima.

PS. b.o.a. stands for basin of attraction(s) in these files.