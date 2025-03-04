#include "seq/sedova_o_min_of_vector_elements/include/ops_seq.hpp"

#include <algorithm>
#include <climits>
#include <random>
#include <vector>

using namespace std::chrono_literals;

std::vector<int> sedova_o_min_of_vector_elements_seq::GetRandomVector(int size, int min, int max) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_int_distribution<> distrib(min, max);
  std::vector<int> vec(size);
  std::generate(vec.begin(), vec.end(), [&]() { return distrib(gen); });
  return vec;
}

std::vector<std::vector<int>> sedova_o_min_of_vector_elements_seq::GetRandomMatrix(int rows, int columns, int min,
                                                                                   int max) {
  std::vector<std::vector<int>> vec(rows);
  std::generate(vec.begin(), vec.end(), [&]() { return GetRandomVector(columns, min, max); });
  return vec;
}

bool sedova_o_min_of_vector_elements_seq::TestTaskSequential::PreProcessingImpl() {
  input_.resize(task_data->inputs_count[0]);
  for (size_t i = 0; i < task_data->inputs_count[0]; i++) {
    int* tmp_ptr = reinterpret_cast<int*>(task_data->inputs[i]);
    input_[i].assign(tmp_ptr, tmp_ptr + task_data->inputs_count[1]);
  }
  res_ = INT_MAX;
  return true;
}

bool sedova_o_min_of_vector_elements_seq::TestTaskSequential::ValidationImpl() {
  return (task_data->inputs_count.size() >= 2) && (task_data->inputs_count[0] > 0 && task_data->inputs_count[1] > 0) &&
         (task_data->outputs_count.size() >= 1) && (task_data->outputs_count[0] == 1);
}

bool sedova_o_min_of_vector_elements_seq::TestTaskSequential::RunImpl() {
  if (input_.empty()) return true;
  res_ = input_[0][0];
  for (const auto& row : input_) {
    for (int val : row) {
      res_ = std::min(res_, val);
    }
  }
  return true;
}

bool sedova_o_min_of_vector_elements_seq::TestTaskSequential::PostProcessingImpl() {
  *reinterpret_cast<int*>(task_data->outputs[0]) = res_;
  return true;
}