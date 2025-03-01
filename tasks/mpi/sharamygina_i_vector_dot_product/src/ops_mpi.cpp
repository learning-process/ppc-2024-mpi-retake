#include "mpi/sharamygina_i_vector_dot_product/include/ops_mpi.h"

#include <gtest/gtest.h>
#include <mpi.h>

#include <boost/mpi.hpp>
#include <random>
#include <thread>
#include <vector>

// #include <boost/mpi/collectives.hpp>
// #include <boost/mpi/collectives/broadcast.hpp>
// #include <boost/mpi/collectives/gatherv.hpp>
// #include <boost/mpi/collectives/scatterv.hpp>
// #include <boost/mpi/communicator.hpp>
// #include <iostream>

bool sharamygina_i_vector_dot_product_mpi::vector_dot_product_mpi::PreProcessingImpl() {
  if (world.rank() == 0) {
    (int)(task_data->inputs_count[0]) < world.size() ? delta = task_data->inputs_count[0]
                                                     : delta = task_data->inputs_count[0] / world.size();
    for (size_t i = 0; i < task_data->inputs.size(); ++i) {
      if (task_data->inputs[i] == nullptr || task_data->inputs_count[i] == 0) {
        return false;
      }
    }
    v1.resize(task_data->inputs_count[0]);
    int* source_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
    std::copy(source_ptr, source_ptr + task_data->inputs_count[0], v1.begin());

    v2.resize(task_data->inputs_count[1]);
    source_ptr = reinterpret_cast<int*>(task_data->inputs[1]);
    std::copy(source_ptr, source_ptr + task_data->inputs_count[1], v2.begin());
  }
  return true;
}

bool sharamygina_i_vector_dot_product_mpi::vector_dot_product_mpi::ValidationImpl() {
  if (world.rank() == 0) {
    if (task_data->inputs.empty() || task_data->outputs.empty() ||
        task_data->inputs_count[0] != task_data->inputs_count[1] || task_data->outputs_count[0] == 0) {
      return false;
    }
  }
  return true;
}

bool sharamygina_i_vector_dot_product_mpi::vector_dot_product_mpi::RunImpl() {
  broadcast(world, delta, 0);
  if (world.rank() == 0) {
    for (int proc = 1; proc < world.size(); ++proc) {
      world.send(proc, 0, v1.data() + proc * delta, delta);
      world.send(proc, 1, v2.data() + proc * delta, delta);
    }
  }
  local_v1.resize(delta);
  local_v2.resize(delta);
  if (world.rank() == 0) {
    std::copy(v1.begin(), v1.begin() + delta, local_v1.begin());
    std::copy(v2.begin(), v2.begin() + delta, local_v2.begin());
  } else {
    world.recv(0, 0, local_v1.data(), delta);
    world.recv(0, 1, local_v2.data(), delta);
  }
  int local_result = 0;
  for (size_t i = 0; i < local_v1.size(); ++i) {
    local_result += local_v1[i] * local_v2[i];
  }
  std::vector<int> full_results;
  gather(world, local_result, full_results, 0);
  res = 0;
  if (world.rank() == 0) {
    for (int result : full_results) {
      res += result;
    }
  }
  if (world.rank() == 0 && (int)(task_data->inputs_count[0]) < world.size()) {
    res = 0;
    for (size_t i = 0; i < v1.size(); ++i) {
      res += v1[i] * v2[i];
    }
  }
  return true;
}

bool sharamygina_i_vector_dot_product_mpi::vector_dot_product_mpi::PostProcessingImpl() {
  if (world.rank() == 0) {
    if (!task_data->outputs.empty()) {
      reinterpret_cast<int*>(task_data->outputs[0])[0] = res;
    } else {
      return false;
    }
  }
  return true;
}