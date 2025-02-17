#include "mpi/khovansky_d_num_of_alternations_signs/include/ops_mpi.hpp"

#include <boost/mpi/collectives.hpp>

bool khovansky_d_num_of_alternations_signs_mpi::NumOfAlternationsSignsSeq::PreProcessingImpl() {
  // Init value for input and output
  unsigned int input_size = task_data->inputs_count[0];
  auto *in_ptr = reinterpret_cast<int *>(task_data->inputs[0]);
  input = std::vector<int>(in_ptr, in_ptr + input_size);

  res = 0;

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
  int input_size = input.size();

  for (int i = 0; i < input_size - 1; i++) {
    if ((input[i] < 0 && input[i + 1] >= 0) || (input[i] >= 0 && input[i + 1] < 0)) {
      res++;
    }
  }

  return true;
}

bool khovansky_d_num_of_alternations_signs_mpi::NumOfAlternationsSignsSeq::PostProcessingImpl() {
  reinterpret_cast<int *>(task_data->outputs[0])[0] = res;

  return true;
}

bool khovansky_d_num_of_alternations_signs_mpi::NumOfAlternationsSignsMpi::PreProcessingImpl() {
  // Init value for input and output
  if (world.rank() == 0) {
    if (!task_data) {
      return false;
    }

    if (task_data->inputs[0] == nullptr && task_data->inputs_count[0] == 0) {
      return false;
    }

    if (task_data->outputs[0] == nullptr) {
      return false;
    }

    int input_size = task_data->inputs_count[0];
    int start_size = input_size / world.size();
    auto *in_ptr = reinterpret_cast<int *>(task_data->inputs[0]);
    input = std::vector<int>(in_ptr, in_ptr + input_size);
    start = std::vector<int>(in_ptr, in_ptr + start_size + uint32_t(world.size() > 1));

    for (int process = 1; process < world.size(); process++) {
      auto local_start = process * start_size;
      auto is_last_proc = (process == world.size() - 1);
      auto size = is_last_proc ? (input_size - local_start) : (start_size + 1);
      world.send(process, 0, std::vector<int>(in_ptr + local_start, in_ptr + local_start + size));
    }
  } else {
    world.recv(0, 0, start);
  }

  res = 0;

  return true;
}

bool khovansky_d_num_of_alternations_signs_mpi::NumOfAlternationsSignsMpi::ValidationImpl() {
  if (world.rank() == 0) {
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
  int process_res = 0;
  int start_size = start.size();
  for (int i = 0; i < start_size - 1; i++) {
    if ((start[i] < 0 && start[i + 1] >= 0) || (start[i] >= 0 && start[i + 1] < 0)) {
      process_res++;
    }
  }
  boost::mpi::reduce(world, process_res, res, std::plus(), 0);
  return true;
}

bool khovansky_d_num_of_alternations_signs_mpi::NumOfAlternationsSignsMpi::PostProcessingImpl() {
  if (world.rank() == 0) {
    reinterpret_cast<int *>(task_data->outputs[0])[0] = res;
  }

  return true;
}