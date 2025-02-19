#include "mpi/strakhov_a_char_freq_counter/include/ops_mpi.hpp"

//  Sequential

bool strakhov_a_char_freq_counter_mpi::CharFreqCounterSeq::PreProcessingImpl() {
  auto *tmp = reinterpret_cast<char *>(task_data->inputs[0]);
  for (size_t i = 0; i < task_data->inputs_count[0]; i++) {
    input_[i] = tmp[i];
  }
  target_ = *reinterpret_cast<char *>(task_data->inputs[1]);
  result_ = 0;
  return true;
}

bool strakhov_a_char_freq_counter_mpi::CharFreqCounterSeq::ValidationImpl() { return (task_data->inputs_count[1] == 1); }

bool strakhov_a_char_freq_counter_mpi::CharFreqCounterSeq::RunImpl() {
  result_ = std::count(input_.begin(), input_.end(), target_);
  return true;
}

bool strakhov_a_char_freq_counter_mpi::CharFreqCounterSeq::PostProcessingImpl() {
  reinterpret_cast<int *>(task_data->outputs[0])[0] = result_;
  return true;
}

//  Parallel

bool strakhov_a_char_freq_counter_mpi::CharFreqCounterPar::PreProcessingImpl() {
  if (world_.rank() == 0) {
    auto *tmp = reinterpret_cast<char *>(task_data->inputs[0]);
    for (size_t i = 0; i < task_data->inputs_count[0]; i++) {
      input_[i] = tmp[i];
    }
    target_ = *reinterpret_cast<char *>(task_data->inputs[1]);
  }

  result_ = 0;
  local_result_ = 0;
  return true;
}

bool strakhov_a_char_freq_counter_mpi::CharFreqCounterPar::ValidationImpl() {
  if (world_.rank() == 0) {
    return (task_data->inputs_count[1] == 1);
  }
  return true;
}

bool strakhov_a_char_freq_counter_mpi::CharFreqCounterPar::RunImpl() {
  int rank_ = world_.rank();
    int input_length_ = task_data->inputs_count[0];  
  int world_size_ = world_.size();
  int segment_ = input_length_ / world_size_;
  int excess_ = input_length_ % world_size_;
  std::vector<int> send_counts_(world_size_, segment_);
  for (int i = 0; i < excess_; i++) {
    send_counts_[i]++;
  }
  std::vector<int> displacements_(world_size_, 0);
  for (int i = 1; i < world_size_; i++) {
    displacements_[i] = displacements_[(i - 1)] + send_counts_[(i - 1)];
  }
  local_input_ = std::vector<int>(send_counts_[rank_]);
  boost::mpi::scatterv(world_, input_.data(), send_counts_, displacements_, local_input_.data(), send_counts_[rank_],
                       0);
  local_result_ = std::count(local_input_.begin(), local_input_.end(), target_);
  reduce(world_, local_result_, result_, std::plus(), 0);
  return true;
}

bool strakhov_a_char_freq_counter_mpi::CharFreqCounterPar::PostProcessingImpl() {
  if (world_.rank() == 0) {
    reinterpret_cast<int *>(task_data->outputs[0])[0] = result_;
  }
  return true;
}
