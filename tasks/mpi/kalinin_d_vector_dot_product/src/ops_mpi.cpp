// Copyright 2024 Nesterov Alexander
#include "mpi/kalinin_d_vector_dot_product/include/ops_mpi.hpp"

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/collectives/broadcast.hpp>
#include <boost/mpi/collectives/reduce.hpp>
#include <cmath>
#include <cstddef>
#include <functional>
#include <vector>
int kalinin_d_vector_dot_product_mpi::VectorDotProduct(const std::vector<int>& v1, const std::vector<int>& v2) {
  long long result = 0;
  for (size_t i = 0; i < v1.size(); i++) {
    result += v1[i] * v2[i];
  }
  return static_cast<int>(result);
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
  res_ = 0;
  return true;
}

bool kalinin_d_vector_dot_product_mpi::TestMPITaskSequential::RunImpl() {
  for (size_t i = 0; i < input_[0].size(); i++) {
    res_ += input_[0][i] * input_[1][i];
  }

  return true;
}

bool kalinin_d_vector_dot_product_mpi::TestMPITaskSequential::PostProcessingImpl() {
  reinterpret_cast<int*>(task_data->outputs[0])[0] = res_;
  return true;
}

bool kalinin_d_vector_dot_product_mpi::TestMPITaskParallel::ValidationImpl() {
  if (world_.rank() == 0) {
    // Check count elements of output
    return (task_data->inputs.size() == task_data->inputs_count.size() && task_data->inputs.size() == 2) &&
           (task_data->inputs_count[0] == task_data->inputs_count[1]) &&
           (task_data->outputs.size() == task_data->outputs_count.size()) && task_data->outputs.size() == 1 &&
           task_data->outputs_count[0] == 1;
  }
  return true;
}

bool kalinin_d_vector_dot_product_mpi::TestMPITaskParallel::PreProcessingImpl() {
  if (world_.rank() == 0) {
    num_processes_ = world_.size();

    input_ = std::vector<std::vector<int>>(task_data->inputs.size());
    for (size_t i = 0; i < input_.size(); i++) {
      auto* tmp_ptr = reinterpret_cast<int*>(task_data->inputs[i]);
      input_[i] = std::vector<int>(task_data->inputs_count[i]);
      for (size_t j = 0; j < task_data->inputs_count[i]; j++) {
        input_[i][j] = tmp_ptr[j];
      }
    }
  }

  res_ = 0;
  return true;
}

bool kalinin_d_vector_dot_product_mpi::TestMPITaskParallel::RunImpl() {
  if (world_.rank() == 0) {
    num_processes_ = world_.size();
  }
  boost::mpi::broadcast(world_, num_processes_, 0);

  int total_elements = 0;
  int delta = 0;
  int remainder = 0;

  if (world_.rank() == 0) {
    total_elements = static_cast<int>(task_data->inputs_count[0]);
    delta = total_elements / num_processes_;
    remainder = total_elements % num_processes_;
  }

  boost::mpi::broadcast(world_, delta, 0);
  boost::mpi::broadcast(world_, remainder, 0);

  counts_.resize(num_processes_);
  for (int i = 0; i < num_processes_; ++i) {
    counts_[i] = delta + (i < remainder ? 1 : 0);
  }

  if (world_.rank() == 0) {
    size_t offset_remainder = counts_[0];
    for (int proc = 1; proc < num_processes_; proc++) {
      size_t current_count = counts_[proc];
      world_.send(proc, 0, input_[0].data() + offset_remainder, static_cast<int>(current_count));
      world_.send(proc, 1, input_[1].data() + offset_remainder, static_cast<int>(current_count));
      offset_remainder += current_count;
    }
  }

  local_input1_ = std::vector<int>(counts_[world_.rank()]);
  local_input2_ = std::vector<int>(counts_[world_.rank()]);

  if (world_.rank() > 0) {
    world_.recv(0, 0, local_input1_.data(), counts_[world_.rank()]);
    world_.recv(0, 1, local_input2_.data(), counts_[world_.rank()]);
  } else {
    local_input1_ = std::vector<int>(input_[0].begin(), input_[0].begin() + counts_[0]);
    local_input2_ = std::vector<int>(input_[1].begin(), input_[1].begin() + counts_[0]);
  }

  int local_res = 0;
  for (size_t i = 0; i < local_input1_.size(); i++) {
    local_res += local_input1_[i] * local_input2_[i];
  }

  boost::mpi::reduce(world_, local_res, res_, std::plus<>(), 0);
  return true;
}
bool kalinin_d_vector_dot_product_mpi::TestMPITaskParallel::PostProcessingImpl() {
  if (world_.rank() == 0) {
    reinterpret_cast<int*>(task_data->outputs[0])[0] = res_;
  }
  return true;
}