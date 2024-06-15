import basin
import numpy as np

h = np.array([
    [4, 2, 2, 2, 2, 4],
    [1, 2, 2, 2, 2, 2],
    [5, 2, 2, 3, 3, 3],
    [1, 1, 2, 3, 4, 3],
    [1, 1, 2, 3, 4, 3],
])

print(h)

arrows = [
    '↓',
    '↘',
    '→',
    '↗',
    '↑',
    '↖',
    '←',
    '↙',
    "@",  # (ab)using index -1 here
]

# calculate the list of maxima and the steepest neighbour for each point
maxima, sn = basin.find_maxima(h)
print(maxima)
print(sn)

label, path_sum = basin.find_basins(h, sn, maxima)

for x in sn:
    for y in x:
        print(arrows[y], end=" ")
    print()

print(label)
print(path_sum)
