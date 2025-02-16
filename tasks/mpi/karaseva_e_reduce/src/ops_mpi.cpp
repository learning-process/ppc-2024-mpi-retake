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

/* void karaseva_e_reduce_mpi::TestTaskMPI::ReduceBinaryTree(boost::mpi::communicator &world) {
  std::vector<int> local_result(size_, 0);

  // Выполняем локальную обработку перед Reduce (частичную сумму)
  for (size_t i = 0; i < size_; ++i) {
    local_result[i] = input_[i] * 2;  // Заглушка
  }

  // Reduce по бинарному дереву
  boost::mpi::reduce(world, local_result.data(), size_, output_.data(), std::plus<int>(), 0);

  if (world.rank() == 0) {
    std::cout << "Reduce completed on rank 0\n";
  }
}*/

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


/*#include "mpi/karaseva_e_reduce/include/ops_mpi.hpp"

#include <cmath>
#include <cstddef>
#include <vector>

bool karaseva_e_reduce_mpi::TestTaskMPI::PreProcessingImpl() {
  // Init value for input and output
  unsigned int input_size = task_data->inputs_count[0];
  auto *in_ptr = reinterpret_cast<int *>(task_data->inputs[0]);
  input_ = std::vector<int>(in_ptr, in_ptr + input_size);

  unsigned int output_size = task_data->outputs_count[0];
  output_ = std::vector<int>(output_size, 0);

  rc_size_ = static_cast<int>(std::sqrt(input_size));
  return true;
}

bool karaseva_e_reduce_mpi::TestTaskMPI::ValidationImpl() {
  // Check equality of counts elements
  return task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool karaseva_e_reduce_mpi::TestTaskMPI::RunImpl() {
  if (world_.rank() == 0) {
    // Multiply matrices
    for (int i = 0; i < rc_size_; ++i) {
      for (int j = 0; j < rc_size_; ++j) {
        for (int k = 0; k < rc_size_; ++k) {
          output_[(i * rc_size_) + j] += input_[(i * rc_size_) + k] * input_[(k * rc_size_) + j];
        }
      }
    }
  } else {
    // Multiply matrices
    for (int j = 0; j < rc_size_; ++j) {
      for (int k = 0; k < rc_size_; ++k) {
        for (int i = 0; i < rc_size_; ++i) {
          output_[(i * rc_size_) + j] += input_[(i * rc_size_) + k] * input_[(k * rc_size_) + j];
        }
      }
    }
  }
  world_.barrier();
  return true;
}

bool karaseva_e_reduce_mpi::TestTaskMPI::PostProcessingImpl() {
  for (size_t i = 0; i < output_.size(); i++) {
    reinterpret_cast<int *>(task_data->outputs[0])[i] = output_[i];
  }
  return true;
}
*/