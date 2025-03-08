#include "mpi/sharamygina_i_vector_dot_product/include/ops_mpi.h"

#include <algorithm>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/collectives/broadcast.hpp>
#include <boost/mpi/collectives/gather.hpp>
#include <vector>

bool sharamygina_i_vector_dot_product_mpi::VectorDotProductMpi::PreProcessingImpl() {
  if (world_.rank() == 0) {
    (int)(task_data->inputs_count[0]) < world_.size() ? delta_ = task_data->inputs_count[0]
                                                      : delta_ = task_data->inputs_count[0] / world_.size();
    for (unsigned int i = 0; i < task_data->inputs.size(); ++i) {
      if (task_data->inputs[i] == nullptr || task_data->inputs_count[i] == 0) {
        return false;
      }
    }
    v1_.resize(task_data->inputs_count[0]);
    int* source_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
    std::copy(source_ptr, source_ptr + task_data->inputs_count[0], v1_.begin());

    v2_.resize(task_data->inputs_count[1]);
    source_ptr = reinterpret_cast<int*>(task_data->inputs[1]);
    std::copy(source_ptr, source_ptr + task_data->inputs_count[1], v2_.begin());
  }
  return true;
}

bool sharamygina_i_vector_dot_product_mpi::VectorDotProductMpi::ValidationImpl() {
  if (world_.rank() == 0) {
    if (task_data->inputs.empty() || task_data->outputs.empty() ||
        task_data->inputs_count[0] != task_data->inputs_count[1] || task_data->outputs_count[0] == 0) {
      return false;
    }
  }
  return true;
}

bool sharamygina_i_vector_dot_product_mpi::VectorDotProductMpi::RunImpl() {
  broadcast(world_, delta_, 0);
  if (world_.rank() == 0) {
    for (int proc = 1; proc < world_.size(); ++proc) {
      world_.send(proc, 0, v1_.data() + (proc * delta_), static_cast<int>(delta_));
      world_.send(proc, 1, v2_.data() + (proc * delta_), static_cast<int>(delta_));
    }
  }
  local_v1_.resize(delta_);
  local_v2_.resize(delta_);
  if (world_.rank() == 0) {
    std::copy(v1_.begin(), v1_.begin() + delta_, local_v1_.begin());
    std::copy(v2_.begin(), v2_.begin() + delta_, local_v2_.begin());
  } else {
    world_.recv(0, 0, local_v1_.data(), static_cast<int>(delta_));
    world_.recv(0, 1, local_v2_.data(), static_cast<int>(delta_));
  }
  int local_result = 0;
  for (unsigned int i = 0; i < local_v1_.size(); ++i) {
    local_result += local_v1_[i] * local_v2_[i];
  }
  std::vector<int> full_results;
  gather(world_, local_result, full_results, 0);
  res_ = 0;
  if (world_.rank() == 0) {
    for (int result : full_results) {
      res_ += result;
    }
  }
  if (world_.rank() == 0 && (int)(task_data->inputs_count[0]) < world_.size()) {
    res_ = 0;
    for (unsigned int i = 0; i < v1_.size(); ++i) {
      res_ += v1_[i] * v2_[i];
    }
  }
  return true;
}

bool sharamygina_i_vector_dot_product_mpi::VectorDotProductMpi::PostProcessingImpl() {
  if (world_.rank() == 0) {
    if (!task_data->outputs.empty()) {
      reinterpret_cast<int*>(task_data->outputs[0])[0] = res_;
    } else {
      return false;
    }
  }
  return true;
}