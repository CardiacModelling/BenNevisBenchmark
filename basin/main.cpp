#include <bits/stdc++.h>
using namespace std;

// size of the matrix
const int N = 5000;
const double NEG_INF = -1e99;
// height matrix
double h[N][N];
// number of rows and columns
int m, n;
// i-th element is the local maximum (-z, x, y) for the i-th basin
vector<tuple<int, int, int>> maxima;
// -1: unlabelled; >=0: the index of the basin
int label[N][N];
const vector<pair<int, int>> neighbors = {
    {-1, 0},
    {1, 0},
    {0, -1},
    {0, 1},
    // uncomment if we want eight neighbors instead of four
    // {-1, -1},
    // {-1, 1},
    // {1, -1},
    // {1, 1},
};

bool valid_coor(int x, int y) {
    // check if a coordinate is valid
    return x >= 1 && x <= m && y >= 1 && y <= n;
}

queue<pair<int, int>> q;
void bfs(int idx, int x, int y) {
    q.push(make_pair(x, y));

    while (!q.empty()) {
        const auto &[x, y] = q.front();
        q.pop();
        label[x][y] = idx;

        for (const auto &[dx, dy] : neighbors) {
            int nx = x + dx;
            int ny = y + dy;
            // visit all valid, unlabelled, lower neighbors
            if (!valid_coor(nx, ny) || label[nx][ny] != -1 || h[nx][ny] > h[x][y]) {
                continue;
            }
            label[nx][ny] = idx;
            q.push(make_pair(nx, ny));
        }
    }
}

int main() {
    freopen("input.txt", "r", stdin);
    memset(label, -1, sizeof(label));

    cin >> m >> n;
    for (int x = 0; x <= m + 1; x++) {
        for (int y = 0; y <= n + 1; y++) {
            if (!valid_coor(x, y)) {
                // make outside the boundaries negative inf
                h[x][y] = NEG_INF;
            } else {
                cin >> h[x][y];
            }
        }
    }

    for (int x = 1; x <= m; x++) {
        for (int y = 1; y <= n; y++) {
            bool local_max = true;
            for (const auto &[dx, dy] : neighbors) {
                int nx = x + dx;
                int ny = y + dy;
                if (h[x][y] < h[nx][ny]) {
                    local_max = false;
                    break;
                }
            }

            if (local_max) {
                maxima.emplace_back(-h[x][y], x, y);
            }
        }
    }

    // make the largest maximum first
    sort(maxima.begin(), maxima.end());

    for (size_t i = 0; i < maxima.size(); i++) {
        const auto &[_, x, y] = maxima[i];
        bfs(i, x, y);

        cout << i << ": (" << x << ", " << y << ")\n";
    }

    for (int x = 1; x <= m; x++) {
        for (int y = 1; y <= n; y++) {
            cout << label[x][y] << " ";
        }
        cout << endl;
    }
}
