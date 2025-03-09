// Copyright 2024 Nesterov Alexander
#include "seq/shishkarev_a_dijkstra_algorithm/include/ops_seq.hpp"

#include <algorithm>
#include <limits>
#include <vector>

void shishkarev_a_dijkstra_algorithm_seq::ConvertToCrs(const std::vector<int>& w, Matrix& matrix, int n) {
  matrix.row_ptr.resize(n + 1);
  int nnz = 0;

  for (int i = 0; i < n; i++) {
    matrix.row_ptr[i] = nnz;
    for (int j = 0; j < n; j++) {
      int weight = w[(i * n) + j];
      if (weight != 0) {
        matrix.values.emplace_back(weight);
        matrix.col_index.emplace_back(j);
        nnz++;
      }
    }
  }
  matrix.row_ptr[n] = nnz;
}

bool shishkarev_a_dijkstra_algorithm_seq::TestTaskSequential::PreProcessingImpl() {
  size_ = static_cast<int>(task_data->inputs_count[1]);
  st_ = static_cast<int>(task_data->inputs_count[2]);

  input_ = std::vector<int>(size_ * size_);
  auto* tmp_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
  input_.assign(tmp_ptr, tmp_ptr + task_data->inputs_count[0]);

  res_ = std::vector<int>(size_, 0);
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

bool shishkarev_a_dijkstra_algorithm_seq::TestTaskSequential::RunImpl() {
  const int inf = std::numeric_limits<int>::max();
  Matrix matrix;
  ConvertToCrs(input_, matrix, size_);

  std::vector<bool> visited(size_, false);
  std::vector<int> d(size_, inf);
  d[st_] = 0;

  for (int i = 0; i < size_; i++) {
    int min = inf;
    int index = -1;
    for (int j = 0; j < size_; j++) {
      if (!visited[j] && d[j] < min) {
        min = d[j];
        index = j;
      }
    }

    if (index == -1) {
      break;
    }

    int u = index;
    visited[u] = true;

    for (int j = matrix.row_ptr[u]; j < matrix.row_ptr[u + 1]; j++) {
      int v = matrix.col_index[j];
      int weight = matrix.values[j];

      if (!visited[v] && d[u] != inf && (d[u] + weight < d[v])) {
        d[v] = d[u] + weight;
      }
    }
  }

  res_ = d;

  return true;
}

bool shishkarev_a_dijkstra_algorithm_seq::TestTaskSequential::PostProcessingImpl() {
  std::copy(res_.begin(), res_.end(), reinterpret_cast<int*>(task_data->outputs[0]));  // NOLINT
  return true;
}