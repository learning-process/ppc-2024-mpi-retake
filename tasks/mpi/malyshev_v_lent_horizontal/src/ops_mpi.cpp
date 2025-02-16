#include "mpi/malyshev_v_lent_horizontal/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/serialization/vector.hpp>
#include <vector>

bool malyshev_lent_horizontal::TestTaskSequential::PreProcessingImpl() {
  InternalOrderTest();

  uint32_t rows = task_data->inputs_count[0];
  uint32_t cols = task_data->inputs_count[1];

  matrix_.resize(rows, std::vector<int32_t>(cols));
  vector_.resize(cols);
  result_.resize(rows);

  int32_t* data;
  for (uint32_t i = 0; i < matrix_.size(); i++) {
    data = reinterpret_cast<int32_t*>(task_data->inputs[i]);
    std::copy(data, data + cols, matrix_[i].data());
  }

  data = reinterpret_cast<int32_t*>(task_data->inputs[rows]);
  std::copy(data, data + cols, vector_.data());

  return true;
}

bool malyshev_lent_horizontal::TestTaskSequential::ValidationImpl() {
  InternalOrderTest();

  uint32_t rows = task_data->inputs_count[0];
  uint32_t cols = task_data->inputs_count[1];
  uint32_t vector_size = task_data->inputs_count[2];

  if (task_data->inputs.size() != rows + 1 || task_data->inputs_count.size() < 3) {
    return false;
  }

  if (cols != vector_size) {
    return false;
  }

  return task_data->outputs_count[0] == task_data->inputs_count[0];
}

bool malyshev_lent_horizontal::TestTaskSequential::RunImpl() {
  InternalOrderTest();

  for (uint32_t i = 0; i < matrix_.size(); i++) {
    result_[i] = 0;
    for (uint32_t j = 0; j < vector_.size(); j++) {
      result_[i] += matrix_[i][j] * vector_[j];
    }
  }

  return true;
}

bool malyshev_lent_horizontal::TestTaskSequential::PostProcessingImpl() {
  InternalOrderTest();

  std::copy(result_.begin(), result_.end(), reinterpret_cast<int32_t*>(task_data->outputs[0]));

  return true;
}

bool malyshev_lent_horizontal::TestTaskParallel::PreProcessingImpl() {
  InternalOrderTest();

  if (world.rank() == 0) {
    uint32_t rows = task_data->inputs_count[0];
    uint32_t cols = task_data->inputs_count[1];

    delta_ = rows / world.size();
    ext_ = rows % world.size();

    matrix_.resize(rows, std::vector<int32_t>(cols));
    vector_.resize(cols);
    result_.resize(rows);

    int32_t* data;
    for (uint32_t i = 0; i < matrix_.size(); i++) {
      data = reinterpret_cast<int32_t*>(task_data->inputs[i]);
      std::copy(data, data + cols, matrix_[i].data());
    }

    data = reinterpret_cast<int32_t*>(task_data->inputs[rows]);
    std::copy(data, data + cols, vector_.data());
  }

  return true;
}

bool malyshev_lent_horizontal::TestTaskParallel::ValidationImpl() {
  InternalOrderTest();

  if (world.rank() == 0) {
    uint32_t rows = task_data->inputs_count[0];
    uint32_t cols = task_data->inputs_count[1];
    uint32_t vector_size = task_data->inputs_count[2];

    if (task_data->inputs.size() != rows + 1 || task_data->inputs_count.size() < 3) {
      return false;
    }

    if (cols != vector_size) {
      return false;
    }

    return task_data->outputs_count[0] == task_data->inputs_count[0];
  }

  return true;
}

bool malyshev_lent_horizontal::TestTaskParallel::RunImpl() {
  InternalOrderTest();

  broadcast(world, delta_, 0);
  broadcast(world, ext_, 0);
  broadcast(world, vector_, 0);

  std::vector<int32_t> sizes(world.size(), delta_);
  for (uint32_t i = 0; i < ext_; i++) {
    sizes[world.size() - i - 1]++;
  }

  local_matrix_.resize(sizes[world.rank()]);
  local_result_.resize(sizes[world.rank()]);

  scatterv(world, matrix_, sizes, local_matrix_.data(), 0);

  for (uint32_t i = 0; i < local_matrix_.size(); i++) {
    local_result_[i] = 0;
    for (uint32_t j = 0; j < vector_.size(); j++) {
      local_result_[i] += local_matrix_[i][j] * vector_[j];
    }
  }

  gatherv(world, local_result_, result_.data(), sizes, 0);

  return true;
}

bool malyshev_lent_horizontal::TestTaskParallel::PostProcessingImpl() {
  InternalOrderTest();

  if (world.rank() == 0) {
    std::copy(result_.begin(), result_.end(), reinterpret_cast<int32_t*>(task_data->outputs[0]));
  }

  return true;
}