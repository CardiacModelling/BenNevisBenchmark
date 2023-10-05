#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <bits/stdc++.h>

namespace py = pybind11;

const double NEG_INF = -1e100;
const std::vector<std::pair<int, int>> neighbors = {
    {-1, 0}, {1, 0}, {0, -1}, {0, 1},
    {-1, -1}, {-1, 1}, {1, -1}, {1, 1},
};

bool valid_coor(int m, int n, int x, int y) {
  return 0 <= x && x < m && 0 <= y && y < n; 
}

void get_maxima(
  double* h, size_t m, size_t n, 
  int* sn, 
  std::vector<std::tuple<double, int, int>>* maxima_vec
) {
  memset(sn, -1, m * n * sizeof(int));

  for (size_t x = 0; x < m; x++) {
    for (size_t y = 0; y < n; y++) {
      bool local_max = true;
      double max_gradient = NEG_INF;

      for (size_t i = 0; i < neighbors.size(); i++) {
        const auto &[dx, dy] = neighbors[i];
        int nx = x + dx;
        int ny = y + dy;
        if (!valid_coor(m, n, nx, ny)) { continue; }
        if (h[x * n + y] < h[nx * n + ny]) {
          local_max = false;
          double current_gradient = (h[nx * n + ny] - h[x * n + y]) / 
            sqrt(dx * dx + dy * dy);
          if (current_gradient > max_gradient) {
            max_gradient = current_gradient;
            sn[x * n + y] = i;
          }
        }
      }
      if (local_max) {
          maxima_vec->emplace_back(-h[x * n + y], x, y);
      }

      if (x % 200 == 0 && y % 200 == 0) std::cout << "."; 
    }
    if (x % 200 == 0) std::cout << "\n";
  }

  std::sort(maxima_vec->begin(), maxima_vec->end());
}

py::tuple find_maxima(py::array_t<double> h_array) {
    // Ensure that the input array is 2-dimensional
    if (h_array.ndim() != 2) {
        throw std::runtime_error("Input array must be 2-dimensional");
    }

    py::buffer_info buf_info = h_array.request();
    size_t m = buf_info.shape[0];
    size_t n = buf_info.shape[1];
    double* h = static_cast<double*>(buf_info.ptr);

    py::array_t<int> sn_array({m, n});
    int* sn = static_cast<int*>(sn_array.request().ptr);

    auto maxima_vec = new std::vector<std::tuple<double, int, int>>;

    get_maxima(h, m, n, sn, maxima_vec);

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

    return py::make_tuple(maxima_array, sn_array);
}


void get_basins(
  double* h, size_t m, size_t n, 
  int* sn, 
  int* maxima, size_t maxima_length, 
  int* label
) {
  memset(label, -1, m * n * sizeof(int));
  auto q = new std::queue<std::pair<int, int>>; 

  for (size_t i = 0; i < maxima_length; i++) {
    int x = maxima[i * 2];
    int y = maxima[i * 2 + 1];
    q->push(std::make_pair(x, y));
    label[x * n + y] = i;

    while (!q->empty()) {
      const auto &[x, y] = q->front(); 
      q->pop();
      for (const auto &[dx, dy]: neighbors) {
        const int xp = x + dx; 
        const int yp = y + dy;
        if (!valid_coor(m, n, xp, yp)) { continue; }
        const int &snp = sn[xp * n + yp];
        if (label[xp * n + yp] != -1 || snp == -1) { continue; }
        // if (x, y) is (xp, xp)'s steepest neighbour
        const auto &[dxp, dyp] = neighbors[snp];
        if (x == xp + dxp && y == yp + dyp) {
          q->push(std::make_pair(xp, yp));
          label[xp * n + yp] = i;
        }
      }
    }

    if (i % 10000 == 0) {
      std::cout << i << " / " << maxima_length << "\n";
    }
  }
}


py::array_t<int> find_basins(
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

  get_basins(h, m, n, sn, maxima, maxima_length, label);

  return label_array;
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
