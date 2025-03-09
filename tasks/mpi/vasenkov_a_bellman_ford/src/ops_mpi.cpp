#include "mpi/vasenkov_a_bellman_ford/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/collectives/all_reduce.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/operations.hpp>
#include <functional>
#include <limits>
#include <vector>

bool vasenkov_a_bellman_ford_mpi::BellmanFordMPI::PreProcessingImpl() {
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

bool vasenkov_a_bellman_ford_mpi::BellmanFordMPI::ValidationImpl() {
  return task_data->inputs_count[0] > 0 && task_data->inputs_count[1] >= 0;
}
bool vasenkov_a_bellman_ford_mpi::BellmanFordMPI::RunImpl() {
  int rank = world_.rank();
  int size = world_.size();

  int active_processes = std::min(size, num_vertices_);
  int vertices_per_process = num_vertices_ / active_processes;
  int remainder = num_vertices_ % active_processes;

  int start_vertex = (rank * vertices_per_process) + std::min(rank, remainder);
  int end_vertex = start_vertex + vertices_per_process + (rank < remainder ? 1 : 0);
  bool is_active = rank < active_processes;

  std::vector<int> temp_distances(num_vertices_);

  for (int i = 0; i < num_vertices_ - 1; ++i) {
    std::ranges::copy(distances_, temp_distances.begin());

    if (is_active) {
      UpdateDistances(start_vertex, end_vertex, temp_distances);
    }

    boost::mpi::all_reduce(world_, temp_distances.data(), num_vertices_, distances_.data(), boost::mpi::minimum<int>());
  }

  bool has_negative_cycle = CheckForNegativeCycles(start_vertex, end_vertex, is_active);

  bool global_has_negative_cycle = false;
  boost::mpi::all_reduce(world_, has_negative_cycle, global_has_negative_cycle, std::logical_or<>());

  return !global_has_negative_cycle;
}

void vasenkov_a_bellman_ford_mpi::BellmanFordMPI::UpdateDistances(int start_vertex, int end_vertex,
                                                                  std::vector<int> &temp_distances) {
  for (int u = start_vertex; u < end_vertex; ++u) {
    for (int j = row_ptr_[u]; j < row_ptr_[u + 1]; ++j) {
      int v = col_ind_[j];
      int weight = weights_[j];
      if (distances_[u] != std::numeric_limits<int>::max() && distances_[u] + weight < temp_distances[v]) {
        temp_distances[v] = distances_[u] + weight;
      }
    }
  }
}

bool vasenkov_a_bellman_ford_mpi::BellmanFordMPI::CheckForNegativeCycles(int start_vertex, int end_vertex,
                                                                         bool is_active) {
  bool has_negative_cycle = false;
  if (is_active) {
    for (int u = start_vertex; u < end_vertex; ++u) {
      for (int j = row_ptr_[u]; j < row_ptr_[u + 1]; ++j) {
        int v = col_ind_[j];
        int weight = weights_[j];
        if (distances_[u] != std::numeric_limits<int>::max() && distances_[u] + weight < distances_[v]) {
          has_negative_cycle = true;
          break;
        }
      }
      if (has_negative_cycle) {
        break;
      }
    }
  }
  return has_negative_cycle;
}

bool vasenkov_a_bellman_ford_mpi::BellmanFordMPI::PostProcessingImpl() {
  if (world_.rank() == 0) {
    auto *out_distances = reinterpret_cast<int *>(task_data->outputs[0]);
    for (int i = 0; i < num_vertices_; ++i) {
      out_distances[i] = distances_[i];
    }
  }
  return true;
}

bool vasenkov_a_bellman_ford_mpi::BellmanFordSequentialMPI::PreProcessingImpl() {
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

bool vasenkov_a_bellman_ford_mpi::BellmanFordSequentialMPI::ValidationImpl() {
  bool result = task_data->inputs_count[0] > 0 && task_data->inputs_count[1] >= 0;

  return result;
}

bool vasenkov_a_bellman_ford_mpi::BellmanFordSequentialMPI::RunImpl() {
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

bool vasenkov_a_bellman_ford_mpi::BellmanFordSequentialMPI::PostProcessingImpl() {
  auto *out_distances = reinterpret_cast<int *>(task_data->outputs[0]);
  for (int i = 0; i < num_vertices_; ++i) {
    out_distances[i] = distances_[i];
  }

  return true;
}