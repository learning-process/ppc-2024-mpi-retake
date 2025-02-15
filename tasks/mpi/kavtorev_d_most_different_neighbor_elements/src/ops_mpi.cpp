#include "mpi/kavtorev_d_most_different_neighbor_elements/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <vector>

#include "seq/kavtorev_d_most_different_neighbor_elements/include/ops_seq.hpp"

bool kavtorev_d_most_different_neighbor_elements_mpi::most_different_neighbor_elements_seq::PreProcessingImpl() {
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

bool kavtorev_d_most_different_neighbor_elements_mpi::most_different_neighbor_elements_seq::ValidationImpl() {
  return task_data->inputs_count[0] > 1 && task_data->outputs_count[0] == 1;
}

bool kavtorev_d_most_different_neighbor_elements_mpi::most_different_neighbor_elements_seq::RunImpl() {
  for (size_t i = 1; i < input_.size(); ++i) {
    if (res.first < input_[i].first) res = input_[i];
  }
  return true;
}

bool kavtorev_d_most_different_neighbor_elements_mpi::most_different_neighbor_elements_seq::PostProcessingImpl() {
  reinterpret_cast<int*>(task_data->outputs[0])[0] = res.first;
  return true;
}

bool kavtorev_d_most_different_neighbor_elements_mpi::most_different_neighbor_elements_mpi::PreProcessingImpl() {
  res = {INT_MIN, -1};
  return true;
}

bool kavtorev_d_most_different_neighbor_elements_mpi::most_different_neighbor_elements_mpi::ValidationImpl() {
  if (world.rank() == 0) {
    return task_data->inputs_count[0] > 1 && task_data->outputs_count[0] == 1;
  }
  return true;
}

bool kavtorev_d_most_different_neighbor_elements_mpi::most_different_neighbor_elements_mpi::RunImpl() {
  int delta_size = 0;
  if (world.rank() == 0) {
    delta_size = (task_data->inputs_count[0]) / world.size();
    size = task_data->inputs_count[0];
    if (task_data->inputs_count[0] % world.size() > 0) delta_size++;
  }
  broadcast(world, delta_size, 0);

  if (world.rank() == 0) {
    input_ = std::vector<int>(world.size() * delta_size + 2, 0);
    auto* tmp = reinterpret_cast<int*>(task_data->inputs[0]);
    std::copy(tmp, tmp + task_data->inputs_count[0], input_.begin());
    for (int proc = 1; proc < world.size(); proc++) {
      world.send(proc, 0, input_.data() + proc * delta_size, delta_size + 1);
    }
  }

  local_input_ = std::vector<int>(delta_size + 1);
  st = world.rank() * delta_size;
  if (world.rank() == 0) {
    local_input_ = std::vector<int>(input_.begin(), input_.begin() + delta_size + 1);
  } else {
    world.recv(0, 0, local_input_.data(), delta_size + 1);
  }

  std::pair<int, int> local_ans = {INT_MIN, -1};
  for (size_t i = 0; (i + st) < size - 1 && i < (local_input_.size() - 1); ++i) {
    std::pair<int, int> tmp = {abs(local_input_[i + 1] - local_input_[i]), i + st};
    local_ans = std::max(local_ans, tmp);
  }
  reduce(world, local_ans, res, boost::mpi::maximum<std::pair<int, int>>(), 0);
  return true;
}

bool kavtorev_d_most_different_neighbor_elements_mpi::most_different_neighbor_elements_mpi::PostProcessingImpl() {
  if (world.rank() == 0) {
    reinterpret_cast<int*>(task_data->outputs[0])[0] = res.first;
  }
  return true;
}