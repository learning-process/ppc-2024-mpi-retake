#include "seq/sedova_o_min_of_vector_elements/include/ops_seq.hpp"

#include <algorithm>
#include <climits>
#include <cstddef>
#include <vector>

using namespace std::chrono_literals;

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
         (!task_data->outputs_count.empty()) && (task_data->outputs_count[0] == 1);
}

bool sedova_o_min_of_vector_elements_seq::TestTaskSequential::RunImpl() {
  std::vector<int> local_res(input_.size());

  for (unsigned int i = 0; i < input_.size(); i++) {
    local_res[i] = *std::ranges::min_element(input_[i].begin(), input_[i].end());
  }

  if (!local_res.empty()) {
    res_ = *std::ranges::min_element(local_res.begin(), local_res.end());
  } else {
    res_ = INT_MAX;
  }
  return true;
}

bool sedova_o_min_of_vector_elements_seq::TestTaskSequential::PostProcessingImpl() {
  *reinterpret_cast<int*>(task_data->outputs[0]) = res_;
  return true;
}