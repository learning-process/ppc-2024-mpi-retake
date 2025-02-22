#include "mpi/khovansky_d_num_of_alternations_signs/include/ops_mpi.hpp"

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/collectives/reduce.hpp>
#include <boost/mpi/communicator.hpp>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

bool khovansky_d_num_of_alternations_signs_mpi::NumOfAlternationsSignsSeq::PreProcessingImpl() {
  // Init value for input and output
  unsigned int input_size = task_data->inputs_count[0];
  auto *in_ptr = reinterpret_cast<int *>(task_data->inputs[0]);
  input_ = std::vector<int>(in_ptr, in_ptr + input_size);

  res_ = 0;

  return true;
}

bool khovansky_d_num_of_alternations_signs_mpi::NumOfAlternationsSignsSeq::ValidationImpl() {
  if (!task_data) {
    return false;
  }

  if (task_data->inputs[0] == nullptr && task_data->inputs_count[0] == 0) {
    return false;
  }

  if (task_data->outputs[0] == nullptr) {
    return false;
  }

  return task_data->inputs_count[0] >= 2 && task_data->outputs_count[0] == 1;
}

bool khovansky_d_num_of_alternations_signs_mpi::NumOfAlternationsSignsSeq::RunImpl() {
  auto input_size = input_.size();

  for (size_t i = 0; i < input_size - 1; i++) {
    if ((input_[i] < 0 && input_[i + 1] >= 0) || (input_[i] >= 0 && input_[i + 1] < 0)) {
      res_++;
    }
  }

  return true;
}

bool khovansky_d_num_of_alternations_signs_mpi::NumOfAlternationsSignsSeq::PostProcessingImpl() {
  reinterpret_cast<int *>(task_data->outputs[0])[0] = res_;

  return true;
}

bool khovansky_d_num_of_alternations_signs_mpi::NumOfAlternationsSignsMpi::PreProcessingImpl() {
  // Init value for input and output
  if (world_.rank() == 0) {
    if (!task_data) {
      return false;
    }

    if (task_data->inputs[0] == nullptr && task_data->inputs_count[0] == 0) {
      return false;
    }

    if (task_data->outputs[0] == nullptr) {
      return false;
    }

    auto input_size = task_data->inputs_count[0];
    auto *in_ptr = reinterpret_cast<int *>(task_data->inputs[0]);
    input_ = std::vector<int>(in_ptr, in_ptr + input_size);
  }

  res_ = 0;

  return true;
}

bool khovansky_d_num_of_alternations_signs_mpi::NumOfAlternationsSignsMpi::ValidationImpl() {
  if (world_.rank() == 0) {
    if (!task_data) {
      return false;
    }

    if (task_data->inputs[0] == nullptr && task_data->inputs_count[0] == 0) {
      return false;
    }

    if (task_data->outputs[0] == nullptr) {
      return false;
    }

    return task_data->inputs_count[0] >= 2 && task_data->outputs_count[0] == 1;
  }

  return true;
}

bool khovansky_d_num_of_alternations_signs_mpi::NumOfAlternationsSignsMpi::RunImpl() {
  if (world_.rank() == 0) {
    auto input_size = task_data->inputs_count[0];
    auto start_size = input_size / world_.size();
    auto *in_ptr = reinterpret_cast<int *>(task_data->inputs[0]);
    input_ = std::vector<int>(in_ptr, in_ptr + input_size);
    start_ = std::vector<int>(in_ptr, in_ptr + start_size + uint32_t(world_.size() > 1));

    for (int process = 1; process < world_.size(); process++) {
      auto local_start = process * start_size;
      auto is_last_proc = (process == world_.size() - 1);
      auto size = is_last_proc ? (input_size - local_start) : (start_size + 1);
      world_.send(process, 0, std::vector<int>(in_ptr + local_start, in_ptr + local_start + size));
    }
  } else {
    world_.recv(0, 0, start_);
  }

  auto process_res = 0;
  auto start_size = start_.size();
  for (size_t i = 0; i < start_size - 1; i++) {
    if ((start_[i] < 0 && start_[i + 1] >= 0) || (start_[i] >= 0 && start_[i + 1] < 0)) {
      process_res++;
    }
  }
  boost::mpi::reduce(world_, process_res, res_, std::plus(), 0);
  return true;
}

bool khovansky_d_num_of_alternations_signs_mpi::NumOfAlternationsSignsMpi::PostProcessingImpl() {
  if (world_.rank() == 0) {
    reinterpret_cast<int *>(task_data->outputs[0])[0] = res_;
  }

  return true;
}