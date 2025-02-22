#include "seq/vasenkov_a_word_count/include/ops_seq.hpp"
#include <iostream>
#include <cmath>
#include <cstddef>
#include <vector>


bool vasenkov_a_word_count_seq::WordCountSequential::PreProcessingImpl() {
  stringSize_ = task_data->inputs_count[0];
  inputString_ = std::string(reinterpret_cast<char*>(task_data->inputs[0]), stringSize_);
  wordCount_ = 0;
  return true;
}

bool vasenkov_a_word_count_seq::WordCountSequential::ValidationImpl() {
  return task_data->outputs_count[0] == 1;
}

bool vasenkov_a_word_count_seq::WordCountSequential::RunImpl() {
  bool inWord = false;
  for (char c : inputString_) {
      if (std::isspace(c)) {
          inWord = false;
      } else if (!inWord) {
        wordCount_++;
          inWord = true;
      }
  }
  return true;
}

bool vasenkov_a_word_count_seq::WordCountSequential::PostProcessingImpl() {
  reinterpret_cast<int *>(task_data->outputs[0])[0] = wordCount_;
  return true;
}
