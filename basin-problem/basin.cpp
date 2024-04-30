#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <bits/stdc++.h>

namespace py = pybind11;

const double NEG_INF = -1e100;
const double EPS = 1e-5;
const std::vector<std::pair<int, int>> neighbors = {
    {1, 0}, 
    {1, 1},
    {0, 1},
    {-1, 1}, 
    {-1, 0}, 
    {-1, -1},
    {0, -1}, 
    {1, -1}, 
};

// check if a coordinate (x, y) is within the valid range [0, m) \times [0, n)
bool valid_coor(int m, int n, int x, int y) {
  return 0 <= x && x < m && 0 <= y && y < n; 
}

double gradient(double* h, size_t n, int x1, int y1, int x2, int y2) {
  int dx = x1 - x2;
  int dy = y1 - y2;
  return (h[x1 * n + y1] - h[x2 * n + y2]) / sqrt(dx * dx + dy * dy);
}
 
void get_maxima(
  // input:
  double* h, size_t m, size_t n, // the height data and size
  // output:
  int* sn, // the index of the steepest neighbour 
  std::vector<std::tuple<double, int, int>>* maxima_vec 
  // the list of local maxima (-height, x, y), 
  // sorted so that the highest appears first
) {
  memset(sn, -1, m * n * sizeof(int));
  auto temp_maxima = new std::vector<std::pair<int, int>>;

  for (size_t x = 0; x < m; x++) {
    for (size_t y = 0; y < n; y++) {
      bool local_max = true;
      double max_gradient = NEG_INF;

      for (size_t i = 0; i < neighbors.size(); i++) {
        const auto &[dx, dy] = neighbors[i];
        // create the coordinate of a neighbour
        int nx = x + dx;
        int ny = y + dy;
        // if it's not within the range forget about it
        if (!valid_coor(m, n, nx, ny)) { continue; }
        // if current height < neighbour's height, 
        // then current point is not a local maximum

        // and we can update the steepest gradient from current point
        // to any of its neighbour

        // note that a local max may = some of its neighbour's height
        if (h[x * n + y] < h[nx * n + ny]) {
          local_max = false;
          double current_gradient = gradient(h, n, nx, ny, x, y);

          if (current_gradient > max_gradient) {
            max_gradient = current_gradient;
            sn[x * n + y] = i;
          }
        }
      }

      // collect all local max in our vector
      if (local_max) {
          temp_maxima->emplace_back(x, y);
      }
      // if (x % 200 == 0 && y % 200 == 0) std::cout << "."; 
    }
    // if (x % 200 == 0) std::cout << "\n";
  }

  // now we go over all the local maxima again to perform the 'sideway moves'
  for (auto &[x, y]: *temp_maxima) {
    double max_gradient = NEG_INF;
    int flat_neighbour_id = -1;

    // we look at each neighbour (nx, ny) of (x, y)
    for (size_t i = 0; i < neighbors.size(); i++) {
      const auto &[dx, dy] = neighbors[i];
      // create the coordinate of a neighbour
      int nx = x + dx;
      int ny = y + dy;
      // if it's not within the range forget about it
      if (!valid_coor(m, n, nx, ny)) { continue; }

      // we want to find a neighbour (nx, ny) which has (almost) the same height as (x, y)
      // such that (nx, ny) is not itself a local max, i.e. it has a steepest neighbour
      if (abs(h[x * n + y] - h[nx * n + ny]) < EPS) {
        if (i <= 3 && flat_neighbour_id == -1) flat_neighbour_id = i;

        if (sn[nx * n + ny] >= 0) {
          // for such a neighbour, we take its steepest neighbour (mx, my)
          // calculate the gradient from (x, y) to (mx, my)
          int j = sn[nx * n + ny];
          int mx = nx + neighbors[j].first;
          int my = ny + neighbors[j].second;

          double current_gradient = gradient(h, n, mx, my, x, y);
          if (current_gradient > max_gradient) {
            max_gradient = current_gradient;

            // we don't want another (x, y) to be using this sn 
            // in this loop we should only use the sn created in the first go
            sn[x * n + y] = i - 100;
          }
        }
      }
    }

    // if we have a unique flat neighbour, which does not have a steepest neighbour
    // we connect to it
    // but we don't want a loop between the two! so the neighbour id must be 0, 1, 2, 3
    if (sn[x * n + y] == -1 && flat_neighbour_id != -1) {
      sn[x * n + y] = flat_neighbour_id - 100;
    }

    // if a local max (x, y) survived all these and its sn didn't get updated
    // it is a *real* local max
    if (sn[x * n + y] == -1) {
      maxima_vec->emplace_back(-h[x * n + y], x, y);
    }
  }

  // finally we recover the sn in a third go
  for (auto &[x, y]: *temp_maxima) {
    if (sn[x * n + y] < -1) sn[x * n + y] += 100;
  }

  // we have used -height, so the highest will appear first after sorting
  std::sort(maxima_vec->begin(), maxima_vec->end());
}

py::tuple find_maxima(py::array_t<double> h_array) {
    // Ensure that the input array is 2-dimensional
    if (h_array.ndim() != 2) {
        throw std::runtime_error("Input array must be 2-dimensional");
    }

    // get data size from input
    py::buffer_info buf_info = h_array.request();
    size_t m = buf_info.shape[0];
    size_t n = buf_info.shape[1];
    double* h = static_cast<double*>(buf_info.ptr); 
    // note that we need to get the underlying array
    // from the py::array_t object to manipulate it

    // init sn array
    py::array_t<int> sn_array({m, n});
    int* sn = static_cast<int*>(sn_array.request().ptr);

    // init vect for local max
    auto maxima_vec = new std::vector<std::tuple<double, int, int>>;

    // lauch the calculation!
    get_maxima(h, m, n, sn, maxima_vec);

    // transfer the local max to a py::array_t
    // we omit the height data because 
    // 1) it saves storage 2) maxima_array can then be an 'int' array
    size_t maxima_length = maxima_vec->size();
    const size_t two = 2;
    py::array_t<int> maxima_array({maxima_length, two});
    int* maxima = static_cast<int*>(maxima_array.request().ptr);
    for (size_t i = 0; i < maxima_length; i++) {
      const auto &[_, x, y] = (*maxima_vec)[i];
      maxima[i * 2] = x;
      maxima[i * 2 + 1] = y;
    }

    delete maxima_vec;
    // maxima_array is a (maxima_length, 2) array containing the x, y coords of 
    // each local maximum, ordered from the highest to the lowest
    // sn_array has the same shape as h, showing the index in [0,1,...,7] of the steepest neighbour
    return py::make_tuple(maxima_array, sn_array);
}


void get_basins(
  // input:
  double* h, size_t m, size_t n, // height data and size
  int* sn, // the steepest_neighbour array
  int* maxima, size_t maxima_length,  // coords of local max and num of them
  // output:
  int* label, // the same shape as h, label[x][y] = the index of local max b.o.a. (x, y) belong to
  int* path_sum // the same shape as maxima, the sum of all paths of this b.o.a.
) {
  // initially all points are unvisited
  memset(label, -1, m * n * sizeof(int));

  // the queue for BFS
  auto q = new std::queue<std::tuple<int, int, int>>; 

  for (size_t i = 0; i < maxima_length; i++) {

    // start from each of the local max
    int x = maxima[i * 2];
    int y = maxima[i * 2 + 1];

    path_sum[i] = 1;

    q->push(std::make_tuple(x, y, 1));
    label[x * n + y] = i;

    while (!q->empty()) {
      const auto &[x, y, path_len] = q->front(); 
      q->pop();
      path_sum[i] += path_len;
      for (const auto &[dx, dy]: neighbors) {
        // expand the neighbours; we are going from high points to low points
        // so we are trying to find a neighbour (xp, yp) such that 
        // (x, y) is the s.n. of (xp, yp)
        const int xp = x + dx; 
        const int yp = y + dy;
        if (!valid_coor(m, n, xp, yp)) { continue; }

        // the s.n. of (xp, yp)
        const int &snp = sn[xp * n + yp];
        
        if (label[xp * n + yp] != -1 || snp == -1) { continue; }
        // if (x, y) is (xp, xp)'s steepest neighbour
        const auto &[dxp, dyp] = neighbors[snp];
        if (x == xp + dxp && y == yp + dyp) {
          q->push(std::make_tuple(xp, yp, path_len + 1));
          label[xp * n + yp] = i;
        }
      }
    }

    if (i % 10000 == 0 || i == maxima_length - 1) {
      std::cout << (i + 1) << " / " << maxima_length << "\n";
    }
  }
}


py::tuple find_basins(
  py::array_t<double> h_array,
  py::array_t<int> sn_array,
  py::array_t<int> maxima_array
) {

  py::buffer_info h_buf = h_array.request();
  size_t m = h_buf.shape[0];
  size_t n = h_buf.shape[1];
  double* h = static_cast<double*>(h_buf.ptr);

  int* sn = static_cast<int*>(sn_array.request().ptr);

  int* maxima = static_cast<int*>(maxima_array.request().ptr);
  size_t maxima_length = maxima_array.request().shape[0];

  py::array_t<int> label_array({m, n});
  int* label = static_cast<int*>(label_array.request().ptr);

  py::array_t<int> path_sum_array(maxima_length);
  int* path_sum = static_cast<int*>(path_sum_array.request().ptr);

  get_basins(h, m, n, sn, maxima, maxima_length, label, path_sum);

  return py::make_tuple(label_array, path_sum_array);
}

py::array_t<int> count_basin_area(
  py::array_t<int> label_array, 
  size_t basin_num, 
  py::array_t<double> h_array,
  bool excluding_sea
) {
  py::array_t<int> area_array(basin_num);
  int* area = static_cast<int*>(area_array.request().ptr);
  memset(area, 0, basin_num * sizeof(int));

  int* label = static_cast<int*>(label_array.request().ptr);
  size_t m = label_array.request().shape[0];
  size_t n = label_array.request().shape[1];

  double* h = static_cast<double*>(h_array.request().ptr);

  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < n; j++) {
      const int &x = label[i * n + j];
      if (!excluding_sea || h[i * n + j] >= 0) {
        area[x]++;
      }
    }
  }

  return area_array;
}

PYBIND11_MODULE(basin, m) {
    m.doc() = "Find all local maxima in a 2D NumPy array";
    m.def("find_maxima", &find_maxima, "Find all local maxima in a 2D NumPy array");
    m.def("find_basins", &find_basins, "Find all basins of attraction.");
    m.def("count_basin_area", &count_basin_area, "Count the area of all basins.");
}
