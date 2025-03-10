// Anikin Maksim 2025
#include "seq/anikin_m_counting_characters/include/ops_seq.hpp"

#include <cmath>
#include <vector>

bool anikin_m_counting_characters_seq::TestTaskSequential::ValidationImpl() {
  return (task_data->inputs.size() == 2) && (task_data->inputs_count.size() == 2) && (task_data->outputs.size() == 1);
}

bool anikin_m_counting_characters_seq::TestTaskSequential::PreProcessingImpl() {
  int input1_size = static_cast<int>(task_data->inputs_count[0]);
  int input2_size = static_cast<int>(task_data->inputs_count[1]);

  res_ = input1_size - input2_size;

  if (res_ <= 0) {
    auto *inlarge_ptr = reinterpret_cast<char *>(task_data->inputs[1]);
    input_1_ = std::vector<char>(inlarge_ptr, inlarge_ptr + input2_size);

    auto *insmall_ptr = reinterpret_cast<char *>(task_data->inputs[0]);
    input_2_ = std::vector<char>(insmall_ptr, insmall_ptr + input1_size);

    res_ = abs(res_);
  } else {
    auto *inlarge_ptr = reinterpret_cast<char *>(task_data->inputs[0]);
    input_1_ = std::vector<char>(inlarge_ptr, inlarge_ptr + input1_size);

    auto *insmall_ptr = reinterpret_cast<char *>(task_data->inputs[1]);
    input_2_ = std::vector<char>(insmall_ptr, insmall_ptr + input2_size);
  }
  return true;
}

bool anikin_m_counting_characters_seq::TestTaskSequential::RunImpl() {
  auto b = input_1_.begin();
  for (auto a : input_2_) {
    if ((a) != (*b)) {
      res_++;
    }
    b++;
  }
  return true;
}

bool anikin_m_counting_characters_seq::TestTaskSequential::PostProcessingImpl() {
  reinterpret_cast<int *>(task_data->outputs[0])[0] = res_;
  return true;
}