// Copyright 2024 Nesterov Alexander
#include "mpi/kalinin_d_vector_dot_product/include/ops_mpi.hpp"

#include <cmath>
#include <cstddef>
#include <vector>

int kalinin_d_vector_dot_product_mpi::vectorDotProduct(const std::vector<int>& v1, const std::vector<int>& v2) {
  long long result = 0;
  for (size_t i = 0; i < v1.size(); i++) result += v1[i] * v2[i];
  return result;
}

bool kalinin_d_vector_dot_product_mpi::TestMPITaskSequential::ValidationImpl() {
  // Check count elements of output
  return (task_data->inputs.size() == task_data->inputs_count.size() && task_data->inputs.size() == 2) &&
         (task_data->inputs_count[0] == task_data->inputs_count[1]) &&
         (task_data->outputs.size() == task_data->outputs_count.size()) && task_data->outputs.size() == 1 &&
         task_data->outputs_count[0] == 1;
}

bool kalinin_d_vector_dot_product_mpi::TestMPITaskSequential::PreProcessingImpl() {
  // Init value for input and output

  input_ = std::vector<std::vector<int>>(task_data->inputs.size());
  for (size_t i = 0; i < input_.size(); i++) {
    auto* tmp_ptr = reinterpret_cast<int*>(task_data->inputs[i]);
    input_[i] = std::vector<int>(task_data->inputs_count[i]);
    for (size_t j = 0; j < task_data->inputs_count[i]; j++) {
      input_[i][j] = tmp_ptr[j];
    }
  }
  res = 0;
  return true;
}

bool kalinin_d_vector_dot_product_mpi::TestMPITaskSequential::RunImpl() {
  for (size_t i = 0; i < input_[0].size(); i++) {
    res += input_[0][i] * input_[1][i];
  }

  return true;
}

bool kalinin_d_vector_dot_product_mpi::TestMPITaskSequential::PostProcessingImpl() {
  reinterpret_cast<int*>(task_data->outputs[0])[0] = res;
  return true;
}

bool kalinin_d_vector_dot_product_mpi::TestMPITaskParallel::ValidationImpl() {
  if (world.rank() == 0) {
    // Check count elements of output
    return (task_data->inputs.size() == task_data->inputs_count.size() && task_data->inputs.size() == 2) &&
           (task_data->inputs_count[0] == task_data->inputs_count[1]) &&
           (task_data->outputs.size() == task_data->outputs_count.size()) && task_data->outputs.size() == 1 &&
           task_data->outputs_count[0] == 1;
  }
  return true;
}

bool kalinin_d_vector_dot_product_mpi::TestMPITaskParallel::PreProcessingImpl() {
  size_t total_elements = 0;
  size_t delta = 0;
  size_t remainder = 0;

  if (world.rank() == 0) {
    total_elements = task_data->inputs_count[0];
    num_processes_ = world.size();
    delta = total_elements / num_processes_;      // Calculate base size for each process
    remainder = total_elements % num_processes_;  // Calculate remaining elements
  }
  boost::mpi::broadcast(world, num_processes_, 0);

  counts_.resize(num_processes_);  // Vector to store counts for each process

  if (world.rank() == 0) {
    // Distribute sizes to each process
    for (unsigned int i = 0; i < num_processes_; ++i) {
      counts_[i] = delta + (i < remainder ? 1 : 0);  // Assign 1 additional element to the first 'remainder' processes
    }
  }
  boost::mpi::broadcast(world, counts_.data(), num_processes_, 0);

  if (world.rank() == 0) {
    input_ = std::vector<std::vector<int>>(task_data->inputs.size());
    for (size_t i = 0; i < input_.size(); i++) {
      auto* tmp_ptr = reinterpret_cast<int*>(task_data->inputs[i]);
      input_[i] = std::vector<int>(task_data->inputs_count[i]);
      for (size_t j = 0; j < task_data->inputs_count[i]; j++) {
        input_[i][j] = tmp_ptr[j];
      }
    }
  }

  res = 0;
  return true;
}

bool kalinin_d_vector_dot_product_mpi::TestMPITaskParallel::RunImpl() {
  if (world.rank() == 0) {
    size_t offset_remainder = counts_[0];
    for (unsigned int proc = 1; proc < num_processes_; proc++) {
      size_t current_count = counts_[proc];
      world.send(proc, 0, input_[0].data() + offset_remainder, current_count);
      world.send(proc, 1, input_[1].data() + offset_remainder, current_count);
      offset_remainder += current_count;
    }
  }

  local_input1_ = std::vector<int>(counts_[world.rank()]);
  local_input2_ = std::vector<int>(counts_[world.rank()]);

  if (world.rank() > 0) {
    world.recv(0, 0, local_input1_.data(), counts_[world.rank()]);
    world.recv(0, 1, local_input2_.data(), counts_[world.rank()]);
  } else {
    local_input1_ = std::vector<int>(input_[0].begin(), input_[0].begin() + counts_[0]);
    local_input2_ = std::vector<int>(input_[1].begin(), input_[1].begin() + counts_[0]);
  }

  int local_res = 0;

  for (size_t i = 0; i < local_input1_.size(); i++) {
    local_res += local_input1_[i] * local_input2_[i];
  }
  boost::mpi::reduce(world, local_res, res, std::plus<>(), 0);
  return true;
}

bool kalinin_d_vector_dot_product_mpi::TestMPITaskParallel::PostProcessingImpl() {
  if (world.rank() == 0) {
    reinterpret_cast<int*>(task_data->outputs[0])[0] = res;
  }
  return true;
}