// Copyright 2023 Nesterov Alexander
#include "mpi/shishkarev_a_sum_of_vector_elements/include/ops_mpi.hpp"

#include <algorithm>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/collectives.hpp>

std::vector<int> shishkarev_a_sum_of_vector_elements_mpi::GetRandomVector(int vector_size) {
  std::mt19937 generator(std::random_device{}());
  std::uniform_int_distribution<int> distribution(0, 99);
  std::vector<int> random_vector(vector_size);
  std::ranges::generate(random_vector, [&]() { return distribution(generator); });  // Исправлено на ranges
  return random_vector;
}

bool shishkarev_a_sum_of_vector_elements_mpi::MPIVectorSumSequential::PreProcessingImpl() {
  input_vector_ = std::vector<int>(task_data->inputs_count[0]);
  int* input_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
  for (unsigned i = 0; i < task_data->inputs_count[0]; i++) {
    input_vector_[i] = input_ptr[i];
  }

  // Инициализация результата
  result_ = 0;
  return true;
}

bool shishkarev_a_sum_of_vector_elements_mpi::MPIVectorSumSequential::ValidationImpl() {
  return task_data->outputs_count[0] == 1;
}

bool shishkarev_a_sum_of_vector_elements_mpi::MPIVectorSumSequential::RunImpl() {
  result_ = std::accumulate(input_vector_.cbegin(), input_vector_.cend(), 0);
  return true;
}

bool shishkarev_a_sum_of_vector_elements_mpi::MPIVectorSumSequential::PostProcessingImpl() {
  *reinterpret_cast<int*>(task_data->outputs[0]) = result_;
  return true;
}

bool shishkarev_a_sum_of_vector_elements_mpi::MPIVectorSumParallel::PreProcessingImpl() {
  int world_id = world.rank();
  int world_size = world.size();
  unsigned int n = 0;

  if (world_id == 0) {
    n = task_data->inputs_count[0];
    input_vector_ = std::vector<int>(n);
    int* input_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
    std::memcpy(input_vector_.data(), input_ptr, sizeof(int) * n);  // Исправлено на std::memcpy
  }

  boost::mpi::broadcast(world, n, 0);

  unsigned int vector_send_size = n / world_size;
  unsigned int local_size = n % world_size;
  std::vector<int> send_counts(world_size, static_cast<int>(vector_send_size));  // Исправлено на static_cast
  std::vector<int> disp(world_size, 0);

  for (unsigned int i = 0; i < static_cast<unsigned int>(world_size); ++i) {
    if (i < local_size) {
      ++send_counts[i];
    }
    if (i > 0) {
      disp[i] = disp[i - 1] + send_counts[i - 1];
    }
  }

  auto local_vector_size = static_cast<unsigned int>(send_counts[world_id]);
  local_vector_.resize(local_vector_size);

  boost::mpi::scatterv(world, input_vector_.data(), send_counts, disp, local_vector_.data(), static_cast<int>(local_vector_size), 0);  // Исправлено на static_cast

  local_sum_ = 0;
  result_ = 0;
  return true;
}

bool shishkarev_a_sum_of_vector_elements_mpi::MPIVectorSumParallel::ValidationImpl() {
  if (world.rank() == 0) {
    return task_data->outputs_count[0] == 1;
  }
  return true;
}

bool shishkarev_a_sum_of_vector_elements_mpi::MPIVectorSumParallel::RunImpl() {
  local_sum_ = std::accumulate(local_vector_.begin(), local_vector_.end(), 0);

  boost::mpi::reduce(world, local_sum_, result_, std::plus<>(), 0);
  return true;
}

bool shishkarev_a_sum_of_vector_elements_mpi::MPIVectorSumParallel::PostProcessingImpl() {
  if (world.rank() == 0) {
    *reinterpret_cast<int*>(task_data->outputs[0]) = result_;
  }

  return true;
}