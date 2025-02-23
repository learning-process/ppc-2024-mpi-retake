
#include "mpi/solovev_a_binary_image_marking/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/collectives/broadcast.hpp>
#include <boost/mpi/collectives/gatherv.hpp>
#include <boost/mpi/collectives/scatterv.hpp>
#include <boost/mpi/communicator.hpp>
#include <iostream>
#include <queue>
#include <vector>

bool solovev_a_binary_image_marking::TestMPITaskSequential::PreProcessingImpl() {
  int m_tmp = *reinterpret_cast<int*>(task_data->inputs[0]);
  int n_tmp = *reinterpret_cast<int*>(task_data->inputs[1]);
  auto* tmp_data = reinterpret_cast<int*>(task_data->inputs[2]);
  data_seq.assign(tmp_data, tmp_data + task_data->inputs_count[2]);
  m_seq = m_tmp;
  n_seq = n_tmp;
  labels_seq.resize(m_seq * n_seq);
  return true;
}

bool solovev_a_binary_image_marking::TestMPITaskSequential::ValidationImpl() {
  int rows_check = *reinterpret_cast<int*>(task_data->inputs[0]);
  int coloms_check = *reinterpret_cast<int*>(task_data->inputs[1]);

  std::vector<int> input_check;

  int* input_check_data = reinterpret_cast<int*>(task_data->inputs[2]);
  int input_check_size = task_data->inputs_count[2];
  input_check.assign(input_check_data, input_check_data + input_check_size);

  return (rows_check > 0 && coloms_check > 0 && !input_check.empty());
}

bool solovev_a_binary_image_marking::TestMPITaskSequential::RunImpl() {
  std::vector<Point> directions = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
  int label = 1;

  std::queue<Point> q;

  for (int i = 0; i < m_seq; ++i) {
    for (int j = 0; j < n_seq; ++j) {
      if (data_seq[i * n_seq + j] == 1 && labels_seq[i * n_seq + j] == 0) {
        q.push({i, j});
        labels_seq[i * n_seq + j] = label;

        while (!q.empty()) {
          Point current = q.front();
          q.pop();

          for (const Point& dir : directions) {
            int newX = current.x + dir.x;
            int newY = current.y + dir.y;

            if (newX >= 0 && newX < m_seq && newY >= 0 && newY < n_seq) {
              int newIdx = newX * n_seq + newY;
              if (data_seq[newIdx] == 1 && labels_seq[newIdx] == 0) {
                labels_seq[newIdx] = label;
                q.push({newX, newY});
              }
            }
          }
        }
        ++label;
      }
    }
  }

  return true;
}

bool solovev_a_binary_image_marking::TestMPITaskSequential::PostProcessingImpl() {
  int* output = reinterpret_cast<int*>(task_data->outputs[0]);
  std::copy(labels_seq.begin(), labels_seq.end(), output);

  return true;
}

bool solovev_a_binary_image_marking::TestMPITaskParallel::PreProcessingImpl() {
  if (world.rank() == 0) {
    int m_count = *reinterpret_cast<int*>(task_data->inputs[0]);
    int n_count = *reinterpret_cast<int*>(task_data->inputs[1]);
    auto* data_tmp = reinterpret_cast<int*>(task_data->inputs[2]);
    data.assign(data_tmp, data_tmp + task_data->inputs_count[2]);
    m = m_count;
    n = n_count;
  }
  return true;
}

bool solovev_a_binary_image_marking::TestMPITaskParallel::ValidationImpl() {
  if (world.rank() == 0) {
    int m_check = *reinterpret_cast<int*>(task_data->inputs[0]);
    int n_check = *reinterpret_cast<int*>(task_data->inputs[1]);
    int input_check_size = task_data->inputs_count[2];
    return (m_check > 0 && n_check > 0 && input_check_size > 0);
  }
  return true;
}

bool solovev_a_binary_image_marking::TestMPITaskParallel::RunImpl() {
  std::vector<Point> directions = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
  boost::mpi::broadcast(world, m, 0);
  boost::mpi::broadcast(world, n, 0);

  int procRank = world.rank();
  int procCount = world.size();

  std::vector<int> counts(procCount, 0);
  std::vector<int> displacements(procCount, 0);

  int currentRowOffset = 0;

  for (int proc = 0; proc < procCount; ++proc) {
    int procRows = m / procCount;
    counts[proc] = procRows * n;
    displacements[proc] = currentRowOffset * n;
    currentRowOffset += procRows;
    if (proc == procCount - 1) counts[proc] += (n * m - (m / procCount) * n * procCount);
  }

  int localPixelCount = counts[procRank];
  std::vector<int> localImage(localPixelCount);
  boost::mpi::scatterv(world, data.data(), counts, displacements, localImage.data(), localPixelCount, 0);
  std::vector<int> localLabels(localPixelCount, 0);
  int baseLabel = displacements[procRank] + 1;
  int currLabel = baseLabel;
  auto toCoordinates = [this](int idx) -> std::pair<int, int> { return {idx / this->n, idx % this->n}; };
  int* pLocalImage = localImage.data();
  int* pLocalLabels = localLabels.data();

  for (int i = 0; i < localPixelCount; ++i) {
    if (pLocalImage[i] == 1 && pLocalLabels[i] == 0) {
      std::queue<Point> bfsQueue;
      auto coord = toCoordinates(i);
      bfsQueue.push({coord.first, coord.second});
      pLocalLabels[i] = currLabel;
      while (!bfsQueue.empty()) {
        Point cp = bfsQueue.front();
        bfsQueue.pop();
        for (const auto& step : directions) {
          int nr = cp.x + step.x, nc = cp.y + step.y;
          if (nr >= 0 && nr < (localPixelCount / n) && nc >= 0 && nc < n) {
            int ni = nr * n + nc;
            if (pLocalImage[ni] == 1 && pLocalLabels[ni] == 0) {
              pLocalLabels[ni] = currLabel;
              bfsQueue.push({nr, nc});
            }
          }
        }
      }
      currLabel++;
    }
  }

  std::vector<int> globalLabels;

  if (procRank == 0) globalLabels.resize(m * n);

  boost::mpi::gatherv(world, localLabels, globalLabels.data(), counts, displacements, 0);

  if (procRank == 0) {
    int total = m * n;
    int* pGlobal = globalLabels.data();
    std::vector<int> parent(total + 1);
    for (int i = 1; i <= total; ++i) parent[i] = i;
    auto findRep = [&parent](int x) -> int {
      while (x != parent[x]) x = parent[x] = parent[parent[x]];
      return x;
    };
    auto unionRep = [&findRep, &parent](int a, int b) {
      int ra = findRep(a), rb = findRep(b);
      if (ra != rb) {
        int newRep = (ra < rb) ? ra : rb;
        int obsolete = (ra < rb) ? rb : ra;
        parent[obsolete] = newRep;
      }
    };
    for (int row = 0; row < m; ++row) {
      for (int col = 0; col < n; ++col) {
        int idx = row * n + col;
        if (pGlobal[idx] == 0) continue;
        int label = pGlobal[idx];
        if (col > 0 && pGlobal[row * n + col - 1] != 0) unionRep(label, pGlobal[row * n + col - 1]);
        if (row > 0 && pGlobal[(row - 1) * n + col] != 0) unionRep(label, pGlobal[(row - 1) * n + col]);
      }
    }
    for (int i = 0; i < total; ++i)
      if (pGlobal[i] != 0) pGlobal[i] = findRep(pGlobal[i]);
    std::unordered_map<int, int> norm;
    int nextLabel = 1;
    for (int i = 0; i < total; ++i) {
      if (pGlobal[i] != 0) {
        int rep = pGlobal[i];
        if (norm.find(rep) == norm.end()) norm[rep] = nextLabel++;
        pGlobal[i] = norm[rep];
      }
    }
    labels = std::move(globalLabels);
  }
  return true;
}

bool solovev_a_binary_image_marking::TestMPITaskParallel::PostProcessingImpl() {
  if (world.rank() == 0) {
    int* output = reinterpret_cast<int*>(task_data->outputs[0]);
    std::copy(labels.begin(), labels.end(), output);
  }
  return true;
}