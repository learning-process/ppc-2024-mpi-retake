
#include "mpi/Shpynov_N_radix_sort/include/Shpynov_N_radix_sort_mpi.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

// *** SEQUENTIAL ***

bool shpynov_n_radix_sort_mpi::TestTaskSEQ::ValidationImpl() {
  if (task_data->inputs_count[0] == 0) {
    return false;
  }
  return task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool shpynov_n_radix_sort_mpi::TestTaskSEQ::PreProcessingImpl() {
  unsigned int input_size = task_data->inputs_count[0];
  auto *in_ptr = reinterpret_cast<int *>(task_data->inputs[0]);
  input_ = std::vector<int>(in_ptr, in_ptr + input_size);

  unsigned int output_size = task_data->outputs_count[0];
  result_ = std::vector<int>(output_size, 0);

  return true;
}

bool shpynov_n_radix_sort_mpi::TestTaskSEQ::RunImpl() {
  result_ = RadixSort(input_);

  return true;
}

bool shpynov_n_radix_sort_mpi::TestTaskSEQ::PostProcessingImpl() {
  auto *output = reinterpret_cast<int *>(task_data->outputs[0]);
  std::ranges::copy(result_.begin(), result_.end(), output);
  return true;
}

// *** PARALLEL ***

bool shpynov_n_radix_sort_mpi::TestTaskMPI::ValidationImpl() {
  if (world_.rank() == 0) {
    if (task_data->inputs_count[0] == 0) {
      return false;
    }
    return task_data->inputs_count[0] == task_data->outputs_count[0];
  }
  return true;
}

bool shpynov_n_radix_sort_mpi::TestTaskMPI::PreProcessingImpl() {
  if (world_.rank() == 0) {
    unsigned int input_size = task_data->inputs_count[0];
    auto *in_ptr = reinterpret_cast<int *>(task_data->inputs[0]);
    input_ = std::vector<int>(in_ptr, in_ptr + input_size);

    unsigned int output_size = task_data->outputs_count[0];
    result_ = std::vector<int>(output_size, 0);
  }

  return true;
}

bool shpynov_n_radix_sort_mpi::TestTaskMPI::RunImpl() {
  size_t InVecSize = input_.size();
  boost::mpi::broadcast(world_, InVecSize, 0);
  std::vector<int> deltas(world_.size(), 0);
  for (int procNum = 0; procNum < world_.size(); procNum++) {
    deltas[procNum] = InVecSize / world_.size();
  }
  for (int i = 0; i < (static_cast<int>(InVecSize) % world_.size()); i++) {
    deltas[i] += 1;
  }
  if (world_.rank() == 0) {
    int deltas_sum = 0;
    for (int procNum = 1; procNum < world_.size(); procNum++) {
      deltas_sum += deltas[procNum - 1];
      world_.send(procNum, 0, input_.data() + deltas_sum, deltas[procNum]);
    }
    std::vector<int> LocVec(input_.begin(), input_.begin() + deltas[0]);
    RadixSort(LocVec);
    result_ = std::move(LocVec);
    for (int i = 1; i < world_.size(); i++) {
      std::vector<int> RecievedSorted(deltas[i]);
      world_.recv(i, 0, RecievedSorted.data(), deltas[i]);
      std::vector<int> MergedPart(result_.size() + RecievedSorted.size());
      std::merge(result_.begin(), result_.end(), RecievedSorted.begin(), RecievedSorted.end(), MergedPart.begin());
      result_ = MergedPart;
    }
  } else {
    std::vector<int> LocVec;
    LocVec.resize((deltas[world_.rank()]));
    world_.recv(0, 0, LocVec.data(), deltas[world_.rank()]);
    RadixSort(LocVec);
    world_.send(0, 0, LocVec.data(), deltas[world_.rank()]);
  }

  return true;
}

bool shpynov_n_radix_sort_mpi::TestTaskMPI::PostProcessingImpl() {
  if (world_.rank() == 0) {
    auto *output = reinterpret_cast<int *>(task_data->outputs[0]);
    std::ranges::copy(result_.begin(), result_.end(), output);
  }
  return true;
}