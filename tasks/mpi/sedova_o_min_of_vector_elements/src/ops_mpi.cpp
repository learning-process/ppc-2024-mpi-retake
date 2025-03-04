#include "mpi/sedova_o_min_of_vector_elements/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/reducer.hpp>
#include <climits>
#include <cmath>
#include <random>
#include <vector>

using namespace std::chrono_literals;

std::vector<int> GetRandomVector(int size, int min, int max) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_int_distribution<> distrib(min, max);
  std::vector<int> vec(size);
  for (int i = 0; i < size; i++) {
    vec[i] = distrib(gen);
  }
  return vec;
}

std::vector<std::vector<int>> GetRandomMatrix(int rows, int columns, int min,
                                                                                   int max) {
  std::vector<std::vector<int>> vec(rows);
  for (int i = 0; i < rows; i++) {
    vec[i] = GetRandomVector(columns, min, max);
  }
  return vec;
}

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
  return (task_data->inputs_count[0] > 0 && task_data->inputs_count[1] > 0) && (task_data->outputs_count[0] == 1);
}

bool sedova_o_min_of_vector_elements_mpi::TestTaskSequential::RunImpl() {
  if (input_.empty()) {
    return true;
  }
  res_ = input_[0][0];
  for (const auto &row : input_) {
    for (int val : row) {
      res_ = std::min(res_, val);
    }
  }
  return true;
}

bool sedova_o_min_of_vector_elements_mpi::TestTaskSequential::PostProcessingImpl() {
  reinterpret_cast<int *>(task_data->outputs[0])[0] = res_;
  return true;
}

bool sedova_o_min_of_vector_elements_mpi::TestTaskMPI::PreProcessingImpl() {
  unsigned int delta = 0;
  if (world_.rank() == 0) {
    delta = task_data->inputs_count[0] * task_data->inputs_count[1] / world_.size();
  }
  boost::mpi::broadcast(world_, delta, 0);
  if (world_.rank() == 0) {
    unsigned int rows = task_data->inputs_count[0];
    unsigned int columns = task_data->inputs_count[1];
    input_ = std::vector<int>(rows * columns);
    for (unsigned int i = 0; i < rows; i++) {
      auto *tmp_ptr = reinterpret_cast<int *>(task_data->inputs[i]);
      for (unsigned int j = 0; j < columns; j++) {
        input_[(i * columns) + j] = tmp_ptr[j];
      }
    }
    for (int proc = 1; proc < world_.size(); proc++) {
      world_.send(proc, 0, input_.data() + (delta * proc), delta);
    }
  }
  output_ = std::vector<int>(delta);
  if (world_.rank() == 0) {
    output_ = std::vector<int>(input_.begin(), input_.begin() + delta);
  } else {
    world_.recv(0, 0, output_.data(), delta);
  }
  res_ = INT_MAX;
  return true;
}

bool sedova_o_min_of_vector_elements_mpi::TestTaskMPI::ValidationImpl() {
  if (world_.rank() == 0) {
    return (task_data->inputs_count[0] > 0 && task_data->inputs_count[1] > 0) && (task_data->outputs_count[0] == 1);
  }
  return true;
}

bool sedova_o_min_of_vector_elements_mpi::TestTaskMPI::RunImpl() {
  int local_res = *std::ranges::min_element(output_.begin(), output_.end());
  boost::mpi::reduce(world_, local_res, res_, boost::mpi::minimum<int>(), 0);
  return true;
}

bool sedova_o_min_of_vector_elements_mpi::TestTaskMPI::PostProcessingImpl() {
  if (world_.rank() == 0) {
    reinterpret_cast<int *>(task_data->outputs[0])[0] = res_;
  }
  return true;
}
