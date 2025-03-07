// Copyright 2024 Nesterov Alexander
#include "seq/shishkarev_a_dijkstra_algorithm/include/ops_seq.hpp"

#include <algorithm>
#include <vector>

const int INF = std::numeric_limits<int>::max();

bool shishkarev_a_dijkstra_algorithm_seq::TestTaskSequential::PreProcessingImpl() {
  // Init value for input and output
  size = task_data->inputs_count[1];
  st = task_data->inputs_count[2];

  input_ = std::vector<int>(size * size);
  auto* tmp_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
  input_.assign(tmp_ptr, tmp_ptr + task_data->inputs_count[0]);

  res_ = std::vector<int>(size, 0);
  return true;
}

bool shishkarev_a_dijkstra_algorithm_seq::TestTaskSequential::ValidationImpl() {
  if (task_data->inputs.empty()) {
    return false;
  }

  if (task_data->inputs_count.size() < 2 || task_data->inputs_count[1] <= 1) {
    return false;
  }

  auto* tmp_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
  if (!std::all_of(tmp_ptr, tmp_ptr + task_data->inputs_count[0], [](int val) { return val >= 0; })) {
    return false;
  }

  if (task_data->inputs_count[2] >= task_data->inputs_count[1]) {
    return false;
  }

  if (task_data->outputs.empty() || task_data->outputs[0] == nullptr || task_data->outputs.size() != 1 ||
      task_data->outputs_count[0] != task_data->inputs_count[1]) {
    return false;
  }

  return true;
}

void shishkarev_a_dijkstra_algorithm_seq::convertToCRS(const std::vector<int>& w, std::vector<int>& values,
                                                       std::vector<int>& colIndex, std::vector<int>& rowPtr, int n) {
  rowPtr.resize(n + 1);
  int nnz = 0;

  for (int i = 0; i < n; i++) {
    rowPtr[i] = nnz;
    for (int j = 0; j < n; j++) {
      int weight = w[i * n + j];
      if (weight != 0) {
        values.emplace_back(weight);
        colIndex.emplace_back(j);
        nnz++;
      }
    }
  }
  rowPtr[n] = nnz;
}

bool shishkarev_a_dijkstra_algorithm_seq::TestTaskSequential::RunImpl() {
  std::vector<int> values;
  std::vector<int> colIndex;
  std::vector<int> rowPtr;
  convertToCRS(input_, values, colIndex, rowPtr, size);

  std::vector<bool> visited(size, false);
  std::vector<int> D(size, INF);
  D[st] = 0;

  for (int i = 0; i < size; i++) {
    int min = INF;
    int index = -1;
    for (int j = 0; j < size; j++) {
      if (!visited[j] && D[j] < min) {
        min = D[j];
        index = j;
      }
    }

    if (index == -1) break;

    int u = index;
    visited[u] = true;

    for (int j = rowPtr[u]; j < rowPtr[u + 1]; j++) {
      int v = colIndex[j];
      int weight = values[j];

      if (!visited[v] && D[u] != INF && (D[u] + weight < D[v])) {
        D[v] = D[u] + weight;
      }
    }
  }

  res_ = D;

  return true;
}

bool shishkarev_a_dijkstra_algorithm_seq::TestTaskSequential::PostProcessingImpl() {
  std::copy(res_.begin(), res_.end(), reinterpret_cast<int*>(task_data->outputs[0]));
  return true;
}