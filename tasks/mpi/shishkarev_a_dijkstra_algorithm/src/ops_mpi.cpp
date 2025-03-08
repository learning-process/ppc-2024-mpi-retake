// Copyright 2023 Nesterov Alexander
#include "mpi/shishkarev_a_dijkstra_algorithm/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/collectives/all_reduce.hpp>
#include <boost/mpi/collectives/broadcast.hpp>
#include <boost/mpi/operations.hpp>
#include <climits>
#include <limits>
#include <utility>
#include <vector>

void shishkarev_a_dijkstra_algorithm_mpi::ConvertToCrs(const std::vector<int>& w, Matrix& matrix, int n) {
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

bool shishkarev_a_dijkstra_algorithm_mpi::TestMPITaskSequential::PreProcessingImpl() {
  size_ = static_cast<int>(task_data->inputs_count[1]);
  st_ = static_cast<int>(task_data->inputs_count[2]);

  input_ = std::vector<int>(size_ * size_);
  auto* tmp_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
  input_.assign(tmp_ptr, tmp_ptr + task_data->inputs_count[0]);

  res_ = std::vector<int>(size_, 0);
  return true;
}

bool shishkarev_a_dijkstra_algorithm_mpi::TestMPITaskSequential::ValidationImpl() {
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

bool shishkarev_a_dijkstra_algorithm_mpi::TestMPITaskSequential::RunImpl() {
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

bool shishkarev_a_dijkstra_algorithm_mpi::TestMPITaskSequential::PostProcessingImpl() {
  std::copy(res_.begin(), res_.end(), reinterpret_cast<int*>(task_data->outputs[0]));  // NOLINT
  return true;
}

bool shishkarev_a_dijkstra_algorithm_mpi::TestMPITaskParallel::PreProcessingImpl() {
  if (world_.rank() == 0) {
    size_ = static_cast<int>(task_data->inputs_count[1]);
    st_ = static_cast<int>(task_data->inputs_count[2]);
  }

  if (world_.rank() == 0) {
    input_ = std::vector<int>(size_ * size_);
    auto* tmp_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
    input_.assign(tmp_ptr, tmp_ptr + task_data->inputs_count[0]);
    Matrix temp_matrix;
    ConvertToCrs(input_, temp_matrix, size_);
    values_ = std::move(temp_matrix.values);
    col_index_ = std::move(temp_matrix.col_index);
    row_ptr_ = std::move(temp_matrix.row_ptr);
  } else {
    input_ = std::vector<int>(size_ * size_, 0);
  }
  return true;
}

bool shishkarev_a_dijkstra_algorithm_mpi::TestMPITaskParallel::ValidationImpl() {
  if (world_.rank() == 0) {
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
  }
  return true;
}

bool shishkarev_a_dijkstra_algorithm_mpi::TestMPITaskParallel::RunImpl() {
  const int inf = std::numeric_limits<int>::max();
  boost::mpi::broadcast(world_, size_, 0);
  boost::mpi::broadcast(world_, st_, 0);

  int values_size = static_cast<int>(values_.size());
  int row_ptr_size = static_cast<int>(row_ptr_.size());
  int col_index_size = static_cast<int>(col_index_.size());

  boost::mpi::broadcast(world_, values_size, 0);
  boost::mpi::broadcast(world_, row_ptr_size, 0);
  boost::mpi::broadcast(world_, col_index_size, 0);

  values_.resize(values_size);
  row_ptr_.resize(row_ptr_size);
  col_index_.resize(col_index_size);

  boost::mpi::broadcast(world_, values_.data(), static_cast<int>(values_.size()), 0);
  boost::mpi::broadcast(world_, row_ptr_.data(), static_cast<int>(row_ptr_.size()), 0);
  boost::mpi::broadcast(world_, col_index_.data(), static_cast<int>(col_index_.size()), 0);

  int delta = size_ / world_.size();
  int extra = size_ % world_.size();
  if (extra != 0) {
    delta += 1;
  }
  int start_index = world_.rank() * delta;
  int end_index = std::min(size_, delta * (world_.rank() + 1));

  res_.resize(size_, INT_MAX);
  std::vector<bool> visited(size_, false);
  std::vector<int> d(size_, inf);

  if (world_.rank() == 0) {
    res_[st_] = 0;
  }

  boost::mpi::broadcast(world_, res_.data(), size_, 0);

  for (int k = 0; k < size_; k++) {
    int local_min = inf;
    int local_index = -1;

    for (int i = start_index; i < end_index; i++) {
      if (!visited[i] && res_[i] < local_min) {
        local_min = res_[i];
        local_index = i;
      }
    }

    std::pair<int, int> local_pair = {local_min, local_index};
    std::pair<int, int> global_pair = {inf, -1};

    boost::mpi::all_reduce(world_, local_pair, global_pair,
                           [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
                             if (a.first < b.first) {
                               return a;
                             }
                             if (a.first > b.first) {
                               return b;
                             }
                             return a;
                           });

    if (global_pair.first == inf || global_pair.second == -1) {
      break;
    }

    visited[global_pair.second] = true;

    for (int j = row_ptr_[global_pair.second]; j < row_ptr_[global_pair.second + 1]; j++) {
      int v = col_index_[j];
      int w = values_[j];

      if (!visited[v] && res_[global_pair.second] != inf && (res_[global_pair.second] + w < res_[v])) {
        res_[v] = res_[global_pair.second] + w;
      }
    }

    boost::mpi::all_reduce(world_, res_.data(), size_, d.data(), boost::mpi::minimum<int>());
    res_ = d;
  }

  return true;
}

bool shishkarev_a_dijkstra_algorithm_mpi::TestMPITaskParallel::PostProcessingImpl() {
  if (world_.rank() == 0) {
    std::copy(res_.begin(), res_.end(), reinterpret_cast<int*>(task_data->outputs[0]));  // NOLINT
  }
  return true;
}