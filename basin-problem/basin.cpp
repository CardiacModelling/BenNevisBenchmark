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
const int FLAG = 100;

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
    }
    if ((x % 200) == 0) std::cout << "Progress of finding local max: " << x << " / " << m << "\n";
  }

  // we have a few rounds of iterations
  int iter = 0;
  while (++iter) {
    bool modified = false;
    std::cout << "Current iteration of expanding local max: " << iter << "\n";
    
    // now we go over all the local maxima again to expand recursively
    for (auto &[x, y]: *temp_maxima) {
      if (sn[x * n + y] != -1) continue;
      double max_gradient = NEG_INF;

      // we look at each neighbour (nx, ny) of (x, y)
      for (size_t i = 0; i < neighbors.size(); i++) {
        const auto &[dx, dy] = neighbors[i];
        // create the coordinate of a neighbour
        int nx = x + dx; int ny = y + dy;
        // if it's not within the range forget about it
        if (!valid_coor(m, n, nx, ny)) { continue; }

        // we want to find a neighbour (nx, ny) which has the same height as (x, y)
        // such that (nx, ny) is not itself a local max, i.e. it has a steepest neighbour
        if (h[x * n + y] == h[nx * n + ny] && sn[nx * n + ny] >= 0) {
          // for such a neighbour, we first find its ((l+1)-th order) steepest neighbour (mx, my)
          // which has a higher height, where l <= iter
          int mx = nx, my = ny;
          for (int l = 1; l <= iter; l++) {
            int j = sn[mx * n + my];
            mx += neighbors[j].first;
            my += neighbors[j].second;
            if (h[mx * n + my] > h[x * n + y]) break;
          }
          if (h[mx * n + my] == h[x * n + y]) continue;

          // calculate the gradient from (x, y) to (mx, my)
          double current_gradient = gradient(h, n, mx, my, x, y);
          if (current_gradient > max_gradient) {
            max_gradient = current_gradient;
            sn[x * n + y] = i;
            modified = true;
          }
        }
      }
    }
    
    // if nothing has been modified in this round,
    // then we have converged
    if (!modified) break;
  }

  
  // now we perform a BFS on all local maxima to connect them
  // if they have the same height and are adjacent
  std::cout << "Now doing BFS to connect flat local maxima... \n"; 
  auto q = new std::queue<std::tuple<int, int>>; 
  for (auto &[x, y]: *temp_maxima) { 
    if (sn[x * n + y] != -1) continue;
    q->push(std::make_tuple(x, y));
    sn[x * n + y] = -2; // modify this value so it does not get connected later

    while (!q->empty()) {
      auto [cx, cy] = q->front();
      q->pop(); 

      // we look at each neighbour (nx, ny) of (x, y)
      for (size_t i = 0; i < neighbors.size(); i++) {
        const auto &[dx, dy] = neighbors[i];
        // create the coordinate of a neighbour
        int nx = cx + dx; int ny = cy + dy;
        // if it's not within the range forget about it
        if (!valid_coor(m, n, nx, ny)) { continue; }
        // if it is not a neighbour with the same height which is also a local maximum,
        // forget about it
        if (h[cx * n + cy] != h[nx * n + ny] || sn[nx * n + ny] != -1) continue;

        // otherwise, we make a connection
        sn[nx * n + ny] = (i + 4) % 8; // the reverse direction!

        q->push(std::make_tuple(nx, ny));
      }
    }
  }


  for (auto &[x, y]: *temp_maxima) { 
    // if a local max (x, y) survived all these
    // then it is a *real* local max
    if (sn[x * n + y] == -2) {
      sn[x * n + y] = -1; // change the value back
      maxima_vec->emplace_back(-h[x * n + y], x, y);
    }
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
  int* path_length = new int[m * n]; 
  auto idx_map = new std::map<std::pair<int, int>, int>;
  for (size_t i = 0; i < maxima_length; i++) { 
    path_sum[i] = 1; 
    int mx = maxima[i * 2];
    int my = maxima[i * 2 + 1];
    (*idx_map)[std::make_pair(mx, my)] = i + 1;
    path_length[mx * n + my] = 1;
    label[mx * n + my] = i;
  }

  auto vec = new std::vector<std::pair<int, int>>;
  for (size_t x = 0; x < m; x++) {
    for (size_t y = 0; y < n; y++) {
      // if this point has been searched before, ignore
      if (label[x * n + y] != -1) { continue; }
      
      vec->clear();
      // we begin from (x, y) and we simply move according to sn
      // whilst recording all the points we visist
      int i = x; int j = y;
      vec->emplace_back(i, j);
      while (sn[i * n + j] != -1 && label[i * n + j] == -1) {
          int t = sn[i * n + j];
          i += neighbors[t].first;
          j += neighbors[t].second;
          vec->emplace_back(i, j);
      }
      // at this point we should either end up at a local maxima (i, j), 
      // or we reach a point which has been labelled before

      // get the index of the local maximum
      int idx = (sn[i * n + j] == -1) ? ((*idx_map)[std::make_pair(i, j)] - 1) : label[i * n + j];

      // this shouldn't happen
      if (idx == -1) {
        std::cerr << "Maxima not included in the array!! " << i << " " << j << "\n";
        continue;
      }

      // delete the last point (which has been labelled already, whether it's a local max or not)
      vec->pop_back();
      // the length of the path from (x, y) to the local maximum
      int l = vec->size() + path_length[i * n + j] ;
      // label all the point recorded with the local maximum index
      for (auto &[px, py]: *vec) {
        path_sum[idx] += l; 
        path_length[px * n + py] = l;
        label[px * n + py] = idx;
        l--;
      }
    }
    if ((x % 200) == 0) std::cout << "Progress of labelling b.o.a.: " << x << " / " << m << "\n";
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

      // this shouldn't happen
      if (!(0 <= x && x < (int)basin_num)) {
        std::cerr << "Invalid label at (i, j) with value x found! where (i, j, x) = (" << i << ", " << j << ", " << x << ")\n";
      }

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
