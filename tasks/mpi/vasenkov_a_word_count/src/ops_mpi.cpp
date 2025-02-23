#include "mpi/vasenkov_a_word_count/include/ops_mpi.hpp"

#include <cmath>
#include <cstddef>
#include <vector>

bool vasenkov_a_word_count_mpi::WordCountMPI::PreProcessingImpl() {
  if (world_.rank() == 0) {
    stringSize_ = task_data->inputs_count[0];
    inputString_ = std::string(reinterpret_cast<char*>(task_data->inputs[0]), stringSize_);
  }
  wordCount_ = 0;
  wordLoaclCount_ = 0;
  return true;
}

bool vasenkov_a_word_count_mpi::WordCountMPI::ValidationImpl() { return task_data->outputs_count[0] != 0; }

bool vasenkov_a_word_count_mpi::WordCountMPI::RunImpl() {
  unsigned int delta = 0;
  if (world_.rank() == 0) {
    delta = inputString_.length() / world_.size();
  }
  broadcast(world_, delta, 0);

  std::string localString;
  localString.resize(delta);

  if (world_.rank() == 0) {
    for (int i = 1; i < world_.size(); i++) {
      world_.send(i, 0, inputString_.data() + (i * delta), delta);
    }
    localString.assign(inputString_.data(), delta);
  } else {
    world_.recv(0, 0, localString.data(), delta);
  }

  bool inWord = false;
  if (world_.rank() != 0) {
    char prevChar;
    world_.recv(world_.rank() - 1, 1, &prevChar, 1);
    inWord = !std::isspace(prevChar);
  }

  wordLoaclCount_ = 0;
  for (char c : localString) {
    if (std::isspace(c)) {
      inWord = false;
    } else if (!inWord) {
      wordLoaclCount_++;
      inWord = true;
    }
  }

  if (world_.rank() != world_.size() - 1) {
    char lastChar = localString.back();
    world_.send(world_.rank() + 1, 1, &lastChar, 1);
  }

  boost::mpi::reduce(world_, wordLoaclCount_, wordCount_, std::plus<>(), 0);

  return true;
}

bool vasenkov_a_word_count_mpi::WordCountMPI::PostProcessingImpl() {
  if (world_.rank() == 0) {
    reinterpret_cast<int*>(task_data->outputs[0])[0] = wordCount_;
  }
  return true;
}
