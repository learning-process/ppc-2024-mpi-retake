#include "mpi/sedova_o_min_of_vector_elements/include/ops_mpi.hpp"
#include <climits>
#include <random>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>
#include <functional>
#include <string>

using namespace std::chrono_literals;


std::vector<int> sedova_o_min_of_vector_elements_mpi::getRandomVector(int size, int min, int max) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_int_distribution<> distrib(min, max);
  std::vector<int> vec(size);
  std::generate(vec.begin(), vec.end(), [&]() { return distrib(gen); });
  return vec;
}

std::vector<std::vector<int>> sedova_o_min_of_vector_elements_mpi::getRandomMatrix(int rows, int columns, int min,
                                                                                   int max) {
  std::vector<std::vector<int>> vec(rows);
  std::generate(vec.begin(), vec.end(), [&]() { return getRandomVector(columns, min, max); });
  return vec;
}

bool sedova_o_min_of_vector_elements_mpi::TestTaskSequential::PreProcessingImpl() {
  input_.resize(task_data->inputs_count[0]);
  for (size_t i = 0; i < task_data->inputs_count[0]; i++) {
    int *tmp_ptr = reinterpret_cast<int *>(task_data->inputs[i]);
    input_[i].assign(tmp_ptr, tmp_ptr + task_data->inputs_count[1]);
  }
  res_ = INT_MAX;
  return true;
}

bool sedova_o_min_of_vector_elements_mpi::TestTaskSequential::ValidationImpl() {
  return (task_data->inputs_count.size() >= 2) && (task_data->inputs_count[0] > 0 && task_data->inputs_count[1] > 0) &&
         (task_data->outputs_count.size() >= 1) && (task_data->outputs_count[0] == 1);
}

bool sedova_o_min_of_vector_elements_mpi::TestTaskSequential::RunImpl() {
  if (input_.empty()) return true;
  res_ = input_[0][0];
  for (const auto &row : input_) {
    for (int val : row) {
      res_ = std::min(res_, val);
    }
  }
  return true;
}

bool sedova_o_min_of_vector_elements_mpi::TestTaskSequential::PostProcessingImpl() {
  *reinterpret_cast<int *>(task_data->outputs[0]) = res_;
  return true;
}

bool sedova_o_min_of_vector_elements_mpi::TestTaskMPI::PreProcessingImpl() {
  if (!ValidationImpl()) {
    return false;
  }

  MPI_Comm_rank(MPI_COMM_WORLD, &world_);
  MPI_Comm_size(MPI_COMM_WORLD, &size_);

  if (size_ <= 0) return false;
  if (world_ == 0) {
    input_.resize(task_data->inputs_count[0]);
    for (size_t i = 0; i < task_data->inputs_count[0]; ++i) {
      int *tmp_ptr = reinterpret_cast<int *>(task_data->inputs[i]);
      input_[i].assign(tmp_ptr, tmp_ptr + task_data->inputs_count[1]);
    }
  }
  return true;
}


bool sedova_o_min_of_vector_elements_mpi::TestTaskMPI::ValidationImpl() {
  if (world_== 0) {
  return (task_data->inputs_count.size() >= 2) && (task_data->inputs_count[0] > 0 && task_data->inputs_count[1] > 0) &&
         (task_data->outputs_count.size() >= 1) && (task_data->outputs_count[0] == 1);
  }
  return true;
}

bool sedova_o_min_of_vector_elements_mpi::TestTaskMPI::RunImpl() {
  int num_rows = task_data->inputs_count[0];
  int num_cols = task_data->inputs_count[1];

  int rows_per_process = num_rows / size_;
  int remainder_rows = num_rows % size_;

  int start_row = world_ * rows_per_process + std::min(world_, remainder_rows);
  int end_row = start_row + rows_per_process + (world_ < remainder_rows ? 1 : 0);

  std::vector<int> local_data;
  if (world_ == 0) {
    for (int i = 1; i < size_; ++i) {
      int start = i * rows_per_process + std::min(i, remainder_rows);
      int count = rows_per_process + (i < remainder_rows ? 1 : 0);
      for (int row_idx = start; row_idx < start + count; ++row_idx) {
        MPI_Send(input_[row_idx].data(), num_cols, MPI_INT, i, row_idx, MPI_COMM_WORLD);
      }
    }
  }
  
  std::vector<std::vector<int>> local_input;
  local_input.resize(end_row - start_row);
  for (int i = 0; i < (end_row - start_row); ++i) {
    local_input[i].resize(num_cols);
    if (world_ == 0) {
      local_input[i] = input_[start_row + i];
    } else {
      MPI_Recv(local_input[i].data(), num_cols, MPI_INT, 0, start_row + i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
  }

  int local_min = INT_MAX;
  for (const auto &row : local_input) {
    for (int val : row) {
      local_min = std::min(local_min, val);
    }
  }

  MPI_Reduce(&local_min, &res_, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);  // Reduce to find min

  return true;
}

bool sedova_o_min_of_vector_elements_mpi::TestTaskMPI::PostProcessingImpl() {
  if (world_ == 0) {
    *reinterpret_cast<int *>(task_data->outputs[0]) = res_;
  }
  return true;
}
