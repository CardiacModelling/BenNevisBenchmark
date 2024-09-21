# %% [markdown]
# ### Plot the b.o.a. labels over the highet map

# %%
import random
import numpy as np
import nevis
import basin


def find_labels(h):
    maxima, sn = basin.find_maxima(h)
    label, _ = basin.find_basins(h, sn, maxima)
    return label, maxima


data = nevis.gb()
vmin = np.min(data)
vmax = np.max(data)

# %% [markdown]
# Let's do the plotting now. We can't plot it on the entire map, so let's
# introduce two methods to truncate the map to a small section.
# 1. The surrounding of a center of the map.
# 2. Based on the square name e.g. NY31


# %%
def get_part(data, center, size):
    """
    Get the the s times s surrounding of the centre (a tuple of index)
    in the 2d matrix data,
    where s = size if size is odd and size+1 if size is even.
    """
    i, j = center
    size //= 2
    # we have to do the maths
    part = data[i - size: i + size + 1,
                j - size: j + size + 1]
    return part


# %%

def get_square(square):
    coords, size = nevis.Coords.from_square_with_size(square)
    x, y = coords.grid
    x //= 50
    y //= 50
    size //= 50
    return data[y:y+size, x:x+size]


def plot_label(h, label=None, maxima=None, show_max_num=None,
               alpha=0.5, seed=12138):
    """
    Plot the label for b.o.a. over the height map.
    h: numpy 2d array for heights
    label: numpy 2d array for b.o.a. labels. Will be calculated if None
    maxima: numpy 1d array for list of local maxima. Will be calculated if None
    show_max_num: the number of the highest local maxima and their b.o.a to be
    shown
    alpha: the transparency of the overlay of the b.o.a.
    seed: used for coloring the b.o.a.
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors

    if label is None or maxima is None:
        label, maxima = find_labels(h)

    def f(x): return (x - vmin) / (vmax - vmin)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        'soundofmusic', [
            (0, '#4872d3'),             # Deep sea blue
            (f(-0.1), '#68b2e3'),       # Shallow sea blue
            (f(0.0), '#0f561e'),        # Dark green
            (f(10), '#1a8b33'),         # Nicer green
            (f(100), '#11aa15'),        # Glorious green
            (f(300), '#e8e374'),        # Yellow at ~1000ft
            (f(610), '#8a4121'),        # Brownish at ~2000ft
            (f(915), '#999999'),        # Grey at ~3000ft
            (1, 'white'),
        ], N=1024)

    plt.imshow(
        h,
        origin='lower',
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation='none',
    )

    if show_max_num is not None:
        label[label >= show_max_num] = show_max_num

    num_colors = len(np.unique(label))
    cmap = plt.get_cmap('gist_ncar', num_colors)
    indices = list(range(num_colors))
    random.seed(seed)
    random.shuffle(indices)
    cmap_object = matplotlib.colors.ListedColormap([cmap(i) for i in indices])

    plt.imshow(
        label,
        origin='lower',
        # cmap='tab20',
        cmap=cmap_object,
        interpolation='none',
        alpha=alpha,
    )

    if show_max_num is not None:
        x, y = maxima[:show_max_num].T
    else:
        x, y = maxima.T

    plt.scatter(y, x, c='purple', s=50, marker='x')
    # plt.scatter(y[1:], x[1:], c='purple', s=50, marker='x')
    # plt.scatter(y[:1], x[:1], c='red', s=50, marker='x')
    plt.show()
    # plt.savefig('out.png')


# %% [markdown]
# Notice: we have calculated the labels for this truncation,
# instead of truncating the labels for the entire map. The
# difference is that there will be some local maxima on the edge
# (as above) that are not local maxima on the entire map.
