#include "mpi/Konstantinov_I_sum_of_vector_elements/include/ops_mpi.hpp"

#include <cmath>
#include <cstddef>
#include <vector>

int Konstantinov_I_sum_of_vector_elements_mpi::vec_elem_sum(const std::vector<int>& vec) {
  int result = 0;
  for (int elem : vec) {
    result += elem;
  }
  return result;
}

bool Konstantinov_I_sum_of_vector_elements_mpi::SumVecElemSequential::PreProcessingImpl() {
  int rows = task_data->inputs_count[0];
  int columns = task_data->inputs_count[1];

  input_ = std::vector<int>(rows * columns);

  for (int i = 0; i < rows; i++) {
    auto* el = reinterpret_cast<int*>(task_data->inputs[i]);
    for (int j = 0; j < columns; j++) {
      input_[i * columns + j] = el[j];
    }
  }

  return true;
}

bool Konstantinov_I_sum_of_vector_elements_mpi::SumVecElemSequential::ValidationImpl() {
  return (task_data->inputs_count.size() == 2 && task_data->inputs_count[0] > 0 && task_data->inputs_count[1] > 0 &&
          task_data->outputs_count.size() == 1 && task_data->outputs_count[0] == 1);
}

bool Konstantinov_I_sum_of_vector_elements_mpi::SumVecElemSequential::RunImpl() {
  result_ = vec_elem_sum(input_);
  return true;
}

bool Konstantinov_I_sum_of_vector_elements_mpi::SumVecElemSequential::PostProcessingImpl() {
  reinterpret_cast<int*>(task_data->outputs[0])[0] = result_;
  return true;
}

bool Konstantinov_I_sum_of_vector_elements_mpi::SumVecElemParallel::PreProcessingImpl() {
  if (world_.rank() == 0) {
    int rows = task_data->inputs_count[0];
    int columns = task_data->inputs_count[1];

    input_ = std::vector<int>(rows * columns);

    for (int i = 0; i < rows; i++) {
      auto* p = reinterpret_cast<int*>(task_data->inputs[i]);
      for (int j = 0; j < columns; j++) {
        input_[i * columns + j] = p[j];
      }
    }
  }

  return true;
}

bool Konstantinov_I_sum_of_vector_elements_mpi::SumVecElemParallel::ValidationImpl() {
  if (world_.rank() == 0)
    return (task_data->inputs_count.size() == 2 && task_data->inputs_count[0] > 0 && task_data->inputs_count[1] > 0 &&
            task_data->outputs_count.size() == 1 && task_data->outputs_count[0] == 1);
  return true;
}

bool Konstantinov_I_sum_of_vector_elements_mpi::SumVecElemParallel::RunImpl() {
  int input_size = 0;
  int local_rank = world_.rank();
  int world_size = world_.size();
  if (local_rank == 0) input_size = input_.size();
  boost::mpi::broadcast(world_, input_size, 0);

  int elem_per_procces = input_size / world_size;
  int residual_elements = input_size % world_size;

  int process_count = elem_per_procces + (local_rank < residual_elements ? 1 : 0);

  std::vector<int> counts(world_size);
  std::vector<int> displacment(world_size);

  for (int i = 0; i < world_size; i++) {
    counts[i] = elem_per_procces + (i < residual_elements ? 1 : 0);
    displacment[i] = i * elem_per_procces + std::min(i, residual_elements);
  }

  output_.resize(counts[local_rank]);
  boost::mpi::scatterv(world_, input_.data(), counts, displacment, output_.data(), process_count, 0);

  int process_sum = vec_elem_sum(output_);
  boost::mpi::reduce(world_, process_sum, result_, std::plus(), 0);

  return true;
}

bool Konstantinov_I_sum_of_vector_elements_mpi::SumVecElemParallel::PostProcessingImpl() {
  if (world_.rank() == 0) reinterpret_cast<int*>(task_data->outputs[0])[0] = result_;
  return true;
}