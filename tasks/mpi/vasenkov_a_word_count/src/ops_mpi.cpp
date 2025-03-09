#include "mpi/vasenkov_a_word_count/include/ops_mpi.hpp"

#include <boost/mpi/collectives/broadcast.hpp>
#include <boost/mpi/collectives/reduce.hpp>
#include <cctype>
#include <functional>

bool vasenkov_a_word_count_mpi::WordCountMPI::PreProcessingImpl() {
  if (world_.rank() == 0) {
    stringSize_ = (int)task_data->inputs_count[0];
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

  std::string local_string;
  local_string.resize(delta);

  if (world_.rank() == 0) {
    for (int i = 1; i < world_.size(); i++) {
      world_.send(i, 0, inputString_.data() + (i * delta), (int)delta);
    }
    local_string.assign(inputString_.data(), delta);
  } else {
    world_.recv(0, 0, local_string.data(), (int)delta);
  }

  bool in_word = false;
  if (world_.rank() != 0) {
    char prev_char = ' ';
    world_.recv(world_.rank() - 1, 1, &prev_char, 1);
    in_word = (std::isspace(prev_char) == 0);
  }

  wordLoaclCount_ = 0;
  for (char c : local_string) {
    if (isspace(c) != 0) {
      in_word = false;
    } else if (!in_word) {
      wordLoaclCount_++;
      in_word = true;
    }
  }

  if (world_.rank() != world_.size() - 1) {
    char last_char = local_string.back();
    world_.send(world_.rank() + 1, 1, &last_char, 1);
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
