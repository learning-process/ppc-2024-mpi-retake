#include "mpi/karaseva_e_reduce/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <functional>
#include <iostream>
#include <numeric>
#include <vector>

bool karaseva_e_reduce_mpi::TestTaskMPI::PreProcessingImpl() {
  if (!task_data || task_data->inputs.empty() || task_data->outputs.empty() || !task_data->inputs[0] ||
      !task_data->outputs[0]) {
    return false;
  }

  unsigned int input_size = task_data->inputs_count[0];
  int *in_ptr = reinterpret_cast<int *>(task_data->inputs[0]);
  input_.assign(in_ptr, in_ptr + input_size);

  unsigned int output_size = task_data->outputs_count[0];
  output_.resize(output_size, 0);  // Гарантируем, что выходной массив имеет нужный размер

  size_ = static_cast<int>(input_size);
  return true;
}

bool karaseva_e_reduce_mpi::TestTaskMPI::ValidationImpl() {
  return task_data && task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool karaseva_e_reduce_mpi::TestTaskMPI::RunImpl() {
  boost::mpi::communicator world;

  if (input_.empty()) {
    return false;
  }

  ReduceBinaryTree(world);
  return true;
}

void karaseva_e_reduce_mpi::TestTaskMPI::ReduceBinaryTree(boost::mpi::communicator &world) {
  int local_sum = std::accumulate(input_.begin(), input_.end(), 0);  // Локальное суммирование
  int global_sum = 0;  // Один элемент для хранения глобальной суммы

  std::cout << "[Rank " << world.rank() << "] Local sum = " << local_sum << std::endl;

  // Выполнение редукции
  boost::mpi::reduce(world, &local_sum, 1, &global_sum, std::plus<int>(), 0);

  if (world.rank() == 0) {
    std::cout << "[Rank 0] Reduce completed, global sum = " << global_sum << std::endl;

    output_ = {global_sum};  // Сохраняем результат в выходной массив
  }
}

bool karaseva_e_reduce_mpi::TestTaskMPI::PostProcessingImpl() {
  if (!task_data || task_data->outputs.empty() || !task_data->outputs[0]) {
    return false;
  }

  int *out_ptr = reinterpret_cast<int *>(task_data->outputs[0]);

  std::cout << "[PostProcessing] Copying output, size = " << output_.size() << std::endl;

  if (output_.size() != task_data->outputs_count[0]) {
    std::cerr << "Output size mismatch!" << std::endl;
    return false;
  }

  // Копируем данные в выходной массив
  std::copy(output_.begin(), output_.end(), out_ptr);

  return true;
}
