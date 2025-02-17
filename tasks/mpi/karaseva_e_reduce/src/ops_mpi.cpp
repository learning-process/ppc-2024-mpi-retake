#include "mpi/karaseva_e_reduce/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <functional>
#include <iostream>
#include <ranges>
#include <vector>

bool karaseva_e_reduce_mpi::TestTaskMPI::PreProcessingImpl() {
  if (!task_data || task_data->inputs.empty() || task_data->outputs.empty()) {
    return false;
  }

  unsigned int input_size = task_data->inputs_count[0];
  auto *in_ptr = reinterpret_cast<int *>(task_data->inputs[0]);
  if (!in_ptr) {
    return false;
  }

  input_.assign(in_ptr, in_ptr + input_size);
  unsigned int output_size = task_data->outputs_count[0];
  output_.assign(output_size, 0);

  size_ = static_cast<int>(input_size);
  return true;
}

bool karaseva_e_reduce_mpi::TestTaskMPI::ValidationImpl() {
  return task_data && task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool karaseva_e_reduce_mpi::TestTaskMPI::RunImpl() {
  boost::mpi::communicator world;
  ReduceBinaryTree(world);
  return true;
}

void karaseva_e_reduce_mpi::TestTaskMPI::ReduceBinaryTree(boost::mpi::communicator &world) {
  boost::mpi::reduce(world, input_.data(), size_, output_.data(), std::plus<>(), 0);

  if (world.rank() == 0) {
    std::cout << "Reduce completed on rank 0\n";
  }
}

bool karaseva_e_reduce_mpi::TestTaskMPI::PostProcessingImpl() {
  if (!task_data || task_data->outputs.empty() || !task_data->outputs[0]) {
    return false;
  }

  auto *out_ptr = reinterpret_cast<int *>(task_data->outputs[0]);
  if (!out_ptr) {
    return false;
  }

  std::ranges::copy(output_, out_ptr);
  return true;
}