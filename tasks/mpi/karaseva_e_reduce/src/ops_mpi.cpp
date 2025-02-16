#include "mpi/karaseva_e_reduce/include/ops_mpi.hpp"

#include <boost/mpi/collectives.hpp>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <numeric>
#include <vector>

bool karaseva_e_reduce_mpi::TestTaskMPI::PreProcessingImpl() {
  unsigned int input_size = task_data->inputs_count[0];
  auto *in_ptr = reinterpret_cast<int *>(task_data->inputs[0]);
  input_.assign(in_ptr, in_ptr + input_size);

  unsigned int output_size = task_data->outputs_count[0];
  output_.assign(output_size, 0);

  size_ = input_size;
  return true;
}

bool karaseva_e_reduce_mpi::TestTaskMPI::ValidationImpl() {
  return task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool karaseva_e_reduce_mpi::TestTaskMPI::RunImpl() {
  boost::mpi::communicator world;
  ReduceBinaryTree(world);
  return true;
}

void karaseva_e_reduce_mpi::TestTaskMPI::ReduceBinaryTree(boost::mpi::communicator &world) {
  boost::mpi::reduce(world, input_.data(), size_, output_.data(), std::plus<int>(), 0);

  if (world.rank() == 0) {
    std::cout << "Reduce completed on rank 0\n";
  }
}

bool karaseva_e_reduce_mpi::TestTaskMPI::PostProcessingImpl() {
  std::copy(output_.begin(), output_.end(), reinterpret_cast<int *>(task_data->outputs[0]));
  return true;
}