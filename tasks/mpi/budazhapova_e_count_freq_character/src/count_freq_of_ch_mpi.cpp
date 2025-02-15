#include <string>
#include <vector>

#include "mpi/budazhapova_e_count_freq_character/include/count_freq_chart_mpi_header.hpp"

int budazhapova_e_count_freq_chart_mpi::counting_freq(std::string str, char symb_) {
  int resalt = 0;
  for (unsigned long i = 0; i < str.length(); i++) {
    if (str[i] == symb_) {
      resalt++;
    }
  }
  return resalt;
}

bool budazhapova_e_count_freq_chart_mpi::TestMPITaskSequential::PreProcessingImpl() {
  InternalOrderTest();
  input_ = std::string(reinterpret_cast<char*>(task_data->inputs[0]), static_cast<int>(task_data->inputs_count[0]));
  symb_ = *reinterpret_cast<char*>(task_data->inputs[1]);
  res_ = 0;
  return true;
}

bool budazhapova_e_count_freq_chart_mpi::TestMPITaskSequential::ValidationImpl() {
  InternalOrderTest();
  return task_data->outputs_count[0] == 1;
}

bool budazhapova_e_count_freq_chart_mpi::TestMPITaskSequential::RunImpl() {
  InternalOrderTest();
  res_ = counting_freq(input_, symb_);
  return true;
}

bool budazhapova_e_count_freq_chart_mpi::TestMPITaskSequential::PostProcessingImpl() {
  InternalOrderTest();
  reinterpret_cast<int*>(task_data->outputs[0])[0] = res_;
  return true;
}

bool budazhapova_e_count_freq_chart_mpi::TestMPITaskParallel::PreProcessingImpl() {
  InternalOrderTest();
  int world_rank = world_.rank();

  if (world_rank == 0) {
    input_ = std::string(reinterpret_cast<char*>(task_data->inputs[0]), static_cast<int>(task_data->inputs_count[0]));
    symb_ = *reinterpret_cast<char*>(task_data->inputs[1]);
  }
  return true;
}

bool budazhapova_e_count_freq_chart_mpi::TestMPITaskParallel::ValidationImpl() {
  InternalOrderTest();
  if (world_.rank() == 0) {
    return task_data->outputs_count[0] == 1;
  }
  return true;
}

bool budazhapova_e_count_freq_chart_mpi::TestMPITaskParallel::RunImpl() {
  InternalOrderTest();
  int world_rank = world_.rank();
  int delta = 0;
  if (world_rank == 0) {
    int input_size = static_cast<int>(task_data->inputs_count[0]);
    delta = (input_size % world_.size() == 0) ? (input_size / world_.size()) : ((input_size / world_.size()) + 1);
  }

  boost::mpi::broadcast(world_, delta, 0);
  boost::mpi::broadcast(world_, symb_, 0);

  if (world_rank == 0) {
    for (int proc = 1; proc < world_.size(); proc++) {
      world_.send(proc, 0, input_.data() + proc * delta, delta);
    }
  }
  local_input_.resize(delta);
  if (world_rank == 0) {
    local_input_ = std::string(input_.begin(), input_.begin() + delta);
  } else {
    world_.recv(0, 0, local_input_.data(), delta);
  }
  local_res = counting_freq(local_input_, symb_);
  boost::mpi::reduce(world_, local_res, res_, std::plus<>(), 0);
  return true;
}

bool budazhapova_e_count_freq_chart_mpi::TestMPITaskParallel::PostProcessingImpl() {
  InternalOrderTest();
  if (world_.rank() == 0) {
    reinterpret_cast<int*>(task_data->outputs[0])[0] = res_;
  }
  return true;
}