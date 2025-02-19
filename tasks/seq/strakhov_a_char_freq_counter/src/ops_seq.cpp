#include "seq/strakhov_a_char_freq_counter/include/ops_seq.hpp"

//  Sequential

bool strakhov_a_char_freq_counter_seq::CharFreqCounterSeq::PreProcessingImpl() {
  auto *tmp = reinterpret_cast<char *>(taskData->inputs[0]);
  for (int i = 0; i < taskData->inputs_count[0]; i++) {
    input_[i] = tmp[i];
  }
  target_ = *reinterpret_cast<char *>(taskData->inputs[1]);
  result_ = 0;
  return true;
}

bool strakhov_a_char_freq_counter_seq::CharFreqCounterSeq::ValidationImpl() { return taskData->inputs_count[1] == 1; }

bool strakhov_a_char_freq_counter_seq::CharFreqCounterSeq::RunImpl() {
  result_ = std::count(input_.begin(), input_.end(), target_);
  return true;
}

bool strakhov_a_char_freq_counter_seq::CharFreqCounterSeq::PostProcessingImpl() {
  reinterpret_cast<int *>(taskData->outputs[0])[0] = result_;
  return true;
}