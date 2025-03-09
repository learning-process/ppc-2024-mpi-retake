
#include "mpi/Shpynov_N_radix_sort/include/Shpynov_N_radix_sort_mpi.hpp"

#include <algorithm>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/collectives/broadcast.hpp>
#include <cmath>
#include <cstddef>
#include <utility>
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
  size_t in_vec_size = input_.size();
  boost::mpi::broadcast(world_, in_vec_size, 0);
  std::vector<int> deltas(world_.size(), 0);
  for (int proc_num = 0; proc_num < world_.size(); proc_num++) {
    deltas[proc_num] = (int)in_vec_size / world_.size();
  }
  for (int i = 0; i < (static_cast<int>(in_vec_size) % world_.size()); i++) {
    deltas[i] += 1;
  }
  if (world_.rank() == 0) {
    int deltas_sum = 0;
    for (int proc_num = 1; proc_num < world_.size(); proc_num++) {
      deltas_sum += deltas[proc_num - 1];
      world_.send(proc_num, 0, input_.data() + deltas_sum, deltas[proc_num]);
    }
    std::vector<int> loc_vec(input_.begin(), input_.begin() + deltas[0]);
    RadixSort(loc_vec);
    result_ = std::move(loc_vec);
    for (int i = 1; i < world_.size(); i++) {
      std::vector<int> recieved_sorted(deltas[i]);
      world_.recv(i, 0, recieved_sorted.data(), deltas[i]);
      std::vector<int> merged_part(result_.size() + recieved_sorted.size());
      std::ranges::merge(result_.begin(), result_.end(), recieved_sorted.begin(), recieved_sorted.end(),
                         merged_part.begin());
      result_ = merged_part;
    }
  } else {
    std::vector<int> loc_vec;
    loc_vec.resize((deltas[world_.rank()]));
    world_.recv(0, 0, loc_vec.data(), deltas[world_.rank()]);
    RadixSort(loc_vec);
    world_.send(0, 0, loc_vec.data(), deltas[world_.rank()]);
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