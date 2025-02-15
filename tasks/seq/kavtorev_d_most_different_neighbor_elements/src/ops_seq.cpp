#include "seq/kavtorev_d_most_different_neighbor_elements/include/ops_seq.hpp"

#include <cmath>
#include <cstddef>
#include <functional>
#include <random>

bool kavtorev_d_most_different_neighbor_elements_seq::most_different_neighbor_elements_seq::PreProcessingImpl() {
  auto input = std::vector<int>(task_data->inputs_count[0]);
  auto* tmp = reinterpret_cast<int*>(task_data->inputs[0]);
  std::copy(tmp, tmp + task_data->inputs_count[0], input.begin());

  input_ = std::vector<std::pair<int, int>>(input.size() - 1);

  for (size_t i = 1; i < input.size(); ++i) {
    input_[i - 1] = {std::abs(input[i] - input[i - 1]), std::min(input[i], input[i - 1])};
  }

  res = input_[0];

  return true;
}

bool kavtorev_d_most_different_neighbor_elements_seq::most_different_neighbor_elements_seq::ValidationImpl() {
  return task_data->inputs_count[0] > 1 && task_data->outputs_count[0] == 1;
}

bool kavtorev_d_most_different_neighbor_elements_seq::most_different_neighbor_elements_seq::RunImpl() {
  for (size_t i = 1; i < input_.size(); ++i) {
    if (res.first < input_[i].first) res = input_[i];
  }

  return true;
}

bool kavtorev_d_most_different_neighbor_elements_seq::most_different_neighbor_elements_seq::PostProcessingImpl() {
  reinterpret_cast<std::pair<int, int>*>(task_data->outputs[0])[0] = {res.second, res.second + res.first};
  return true;
}
