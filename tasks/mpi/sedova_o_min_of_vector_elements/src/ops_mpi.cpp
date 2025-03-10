#include "mpi/sedova_o_min_of_vector_elements/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/collectives/broadcast.hpp>
#include <boost/mpi/collectives/reduce.hpp>
#include <boost/mpi/communicator.hpp>
#include <climits>
#include <vector>

template <typename T>
struct Minimum {
  using FirstArgumentType = T;
  using SecondArgumentType = T;
  using ResultType = T;
  const T &operator()(const T &x, const T &y) const { return x < y ? x : y; }
};

bool sedova_o_min_of_vector_elements_mpi::TestTaskSequential::PreProcessingImpl() {
  input_ = std::vector<std::vector<int>>(task_data->inputs_count[0], std::vector<int>(task_data->inputs_count[1]));
  for (unsigned int i = 0; i < task_data->inputs_count[0]; i++) {
    auto *tmp_ptr = reinterpret_cast<int *>(task_data->inputs[i]);
    std::copy(tmp_ptr, tmp_ptr + task_data->inputs_count[1], input_[i].begin());
  }

  res_ = INT_MAX;
  return true;
}

bool sedova_o_min_of_vector_elements_mpi::TestTaskSequential::ValidationImpl() {
  return task_data->outputs_count[0] == 1 && task_data->inputs_count[0] > 0 && task_data->inputs_count[1] > 0;
}

bool sedova_o_min_of_vector_elements_mpi::TestTaskSequential::RunImpl() {
  std::vector<int> local_res(input_.size());

  for (unsigned int i = 0; i < input_.size(); i++) {
    local_res[i] = *std::min_element(input_[i].begin(), input_[i].end());
  }

  res_ = *std::min_element(local_res.begin(), local_res.end());
  return true;
}

bool sedova_o_min_of_vector_elements_mpi::TestTaskSequential::PostProcessingImpl() {
  reinterpret_cast<int *>(task_data->outputs[0])[0] = res_;
  return true;
}

bool sedova_o_min_of_vector_elements_mpi::TestTaskMPI::PreProcessingImpl() {
  res_ = INT_MAX;
  return true;
}

bool sedova_o_min_of_vector_elements_mpi::TestTaskMPI::ValidationImpl() {
  if (world_.rank() == 0) {
    return (task_data->outputs_count[0] == 1 && (task_data->inputs_count[0] > 0));
  }
  return true;
}

bool sedova_o_min_of_vector_elements_mpi::TestTaskMPI::RunImpl() {
  unsigned int delta = 0;
  if (world_.rank() == 0) {
    delta = task_data->inputs_count[0] * task_data->inputs_count[1] / world_.size();
  }
  boost::mpi::broadcast(world_, delta, 0);
  if (world_.rank() == 0) {
    unsigned int rows_ = task_data->inputs_count[0];
    unsigned int cols_ = task_data->inputs_count[1];
    input_ = std::vector<int>(rows_ * cols_);
    for (unsigned int i = 0; i < rows_; ++i) {
      auto *tmp_ptr = reinterpret_cast<int *>(task_data->inputs[i]);
      for (unsigned int j = 0; j < cols_; ++j) {
        input_[(i * cols_) + j] = tmp_ptr[j];
      }
    }
    for (int proc = 1; proc < world_.size(); ++proc) {
      world_.send(proc, 0, input_.data() + (delta * proc), delta);
    }
  }
  output_ = std::vector<int>(delta);
  if (world_.rank() == 0) {
    output_ = std::vector<int>(input_.begin(), input_.begin() + delta);
  } else {
    world_.recv(0, 0, output_.data(), delta);
  }
  int local_res = INT_MAX;
  if (!local_res.empty()) {
    res_ = *std::min_element(local_res.begin(), local_res.end());
  } else {
    res_ = INT_MAX;
  }
  return true;
}

bool sedova_o_min_of_vector_elements_mpi::TestTaskMPI::PostProcessingImpl() {
  if (world_.rank() == 0) {
    reinterpret_cast<int *>(task_data->outputs[0])[0] = res_;
  }
  return true;
}
