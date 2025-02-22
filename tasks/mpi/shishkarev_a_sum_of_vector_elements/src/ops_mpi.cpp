// Copyright 2023 Nesterov Alexander
#include "mpi/shishkarev_a_sum_of_vector_elements/include/ops_mpi.hpp"

#include <algorithm>
#include <cstring>  // Добавлено для memcpy
#include <functional>
#include <numeric>
#include <random>
#include <vector>

#include <boost/mpi/collectives.hpp>  // Добавлено для broadcast, scatterv, reduce

std::vector<int> shishkarev_a_sum_of_vector_elements_mpi::getRandomVector(int vector_size) {
  std::mt19937 generator(std::random_device{}());
  std::uniform_int_distribution<int> distribution(0, 99);
  std::vector<int> random_vector(vector_size);
  std::ranges::generate(random_vector, [&]() { return distribution(generator); });  // Исправлено на ranges
  return random_vector;
}

bool shishkarev_a_sum_of_vector_elements_mpi::MPIVectorSumSequential::PreProcessingImpl() {
  input_vector = std::vector<int>(task_data->inputs_count[0]);
  int* input_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
  for (unsigned i = 0; i < task_data->inputs_count[0]; i++) {
    input_vector[i] = input_ptr[i];
  }

  // Инициализация результата
  result = 0;
  return true;
}

bool shishkarev_a_sum_of_vector_elements_mpi::MPIVectorSumSequential::ValidationImpl() {
  return task_data->outputs_count[0] == 1;
}

bool shishkarev_a_sum_of_vector_elements_mpi::MPIVectorSumSequential::RunImpl() {
  result = std::accumulate(input_vector.cbegin(), input_vector.cend(), 0);
  return true;
}

bool shishkarev_a_sum_of_vector_elements_mpi::MPIVectorSumSequential::PostProcessingImpl() {
  *reinterpret_cast<int*>(task_data->outputs[0]) = result;
  return true;
}

bool shishkarev_a_sum_of_vector_elements_mpi::MPIVectorSumParallel::PreProcessingImpl() {
  int world_id = world.rank();
  int world_size = world.size();
  unsigned int n = 0;

  if (world_id == 0) {
    n = task_data->inputs_count[0];
    input_vector = std::vector<int>(n);
    int* input_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
    std::memcpy(input_vector.data(), input_ptr, sizeof(int) * n);  // Исправлено на std::memcpy
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
  local_vector.resize(local_vector_size);

  boost::mpi::scatterv(world, input_vector.data(), send_counts, disp, local_vector.data(), static_cast<int>(local_vector_size), 0);  // Исправлено на static_cast

  local_sum = 0;
  result = 0;
  return true;
}

bool shishkarev_a_sum_of_vector_elements_mpi::MPIVectorSumParallel::ValidationImpl() {
  if (world.rank() == 0) {
    return task_data->outputs_count[0] == 1;
  }
  return true;
}

bool shishkarev_a_sum_of_vector_elements_mpi::MPIVectorSumParallel::RunImpl() {
  local_sum = std::accumulate(local_vector.begin(), local_vector.end(), 0);

  boost::mpi::reduce(world, local_sum, result, std::plus<>(), 0);
  return true;
}

bool shishkarev_a_sum_of_vector_elements_mpi::MPIVectorSumParallel::PostProcessingImpl() {
  if (world.rank() == 0) {
    *reinterpret_cast<int*>(task_data->outputs[0]) = result;
  }

  return true;
}