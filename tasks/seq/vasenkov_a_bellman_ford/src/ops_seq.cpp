#include "seq/vasenkov_a_bellman_ford/include/ops_seq.hpp"

#include <iostream>
#include <limits>
#include <vector>

bool vasenkov_a_bellman_ford_seq::BellmanFordSequential::PreProcessingImpl() {
  auto *in_row_ptr = reinterpret_cast<int *>(task_data->inputs[0]);
  auto *in_col_ind = reinterpret_cast<int *>(task_data->inputs[1]);
  auto *in_weights = reinterpret_cast<int *>(task_data->inputs[2]);
  num_vertices_ = *reinterpret_cast<int *>(task_data->inputs[3]);
  source_vertex_ = *reinterpret_cast<int *>(task_data->inputs[4]);

  row_ptr_ = std::vector<int>(in_row_ptr, in_row_ptr + num_vertices_ + 1);
  col_ind_ = std::vector<int>(in_col_ind, in_col_ind + row_ptr_[num_vertices_]);
  weights_ = std::vector<int>(in_weights, in_weights + row_ptr_[num_vertices_]);

  distances_ = std::vector<int>(num_vertices_, std::numeric_limits<int>::max());
  distances_[source_vertex_] = 0;

  return true;
}

bool vasenkov_a_bellman_ford_seq::BellmanFordSequential::ValidationImpl() {
  return task_data->inputs_count[0] > 0 && task_data->inputs_count[1] >= 0;
}

bool vasenkov_a_bellman_ford_seq::BellmanFordSequential::RunImpl() {
  for (int i = 0; i < num_vertices_ - 1; ++i) {
    for (int u = 0; u < num_vertices_; ++u) {
      for (int j = row_ptr_[u]; j < row_ptr_[u + 1]; ++j) {
        int v = col_ind_[j];
        int weight = weights_[j];
        if (distances_[u] != std::numeric_limits<int>::max() && distances_[u] + weight < distances_[v]) {
          distances_[v] = distances_[u] + weight;
        }
      }
    }
  }

  for (int u = 0; u < num_vertices_; ++u) {
    for (int j = row_ptr_[u]; j < row_ptr_[u + 1]; ++j) {
      int v = col_ind_[j];
      int weight = weights_[j];
      if (distances_[u] != std::numeric_limits<int>::max() && distances_[u] + weight < distances_[v]) {
        return false;
      }
    }
  }
  return true;
}

bool vasenkov_a_bellman_ford_seq::BellmanFordSequential::PostProcessingImpl() {
  auto *out_distances = reinterpret_cast<int *>(task_data->outputs[0]);
  for (int i = 0; i < num_vertices_; ++i) {
    out_distances[i] = distances_[i];
  }
  return true;
}