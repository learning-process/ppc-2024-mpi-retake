#include "seq/chernova_n_word_count/include/ops_seq.hpp"

#include <iostream>
#include <string>
#include <thread>
#include <vector>

std::vector<char> clean_string(const std::vector<char>& input) {
  std::string result;
  std::string str(input.begin(), input.end());

  std::string::size_type pos = 0;
  while ((pos = str.find("  ", pos)) != std::string::npos) {
    str.erase(pos, 1);
  }

  pos = 0;
  while ((pos = str.find(" - ", pos)) != std::string::npos) {
    str.erase(pos, 2);
  }

  pos = 0;
  if (str[pos] == ' ') {
    str.erase(pos, 1);
  }

  pos = str.size() - 1;
  if (str[pos] == ' ') {
    str.erase(pos, 1);
  }

  result.assign(str.begin(), str.end());
  return std::vector<char>(result.begin(), result.end());
}

bool chernova_n_word_count_seq::TestTaskSequential::PreProcessingImpl() {
  // Init value for input
  unsigned int input_size = task_data->inputs_count[0];
  auto* in_ptr = reinterpret_cast<char*>(task_data->inputs[0]);
  input_ = std::vector<char>(in_ptr, in_ptr + input_size);

  // Clean the input string
  input_ = clean_string(input_);
  spaceCount = 0;

  return true;
}

bool chernova_n_word_count_seq::TestTaskSequential::ValidationImpl() {
  // Check if input and output counts are valid
  return task_data->inputs_count[0] >= 0 && task_data->outputs_count[0] == 1;
}

bool chernova_n_word_count_seq::TestTaskSequential::RunImpl() {
  // Count spaces in the input string
  if (input_.empty()) {
    spaceCount = -1;
  } else {
    for (char c : input_) {
      if (c == ' ') {
        spaceCount++;
      }
    }
  }
  return true;
}

bool chernova_n_word_count_seq::TestTaskSequential::PostProcessingImpl() {
  // Set the output value
  reinterpret_cast<int*>(task_data->outputs[0])[0] = spaceCount + 1;
  return true;
}