#include "mpi/fomin_v_sentence_count/include/ops_mpi.hpp"

#include <boost/mpi.hpp>
#include <boost/mpi/communicator.hpp>
#include <cstring>
#include <vector>

using namespace std::chrono_literals;

bool SentenceCountParallel::PreProcessingImpl() {
  int world_rank = world.rank();

  if (world_rank == 0) {
    char *input_ptr = reinterpret_cast<char *>(task_data->inputs[0]);
    input_size = taskData->inputs_count[0];
    input_vec.assign(input_ptr, input_ptr + input_size);
  }

  boost::mpi::broadcast(world, input_size, 0);

  portion_size = input_size / world.size();
  int remainder = input_size % world.size();
  if (world_rank < remainder) {
    portion_size++;
  }

  local_input_vec.resize(portion_size);

  std::vector<int> recv_counts(world.size());
  std::vector<int> displs(world.size());

  for (int i = 0; i < world.size(); ++i) {
    recv_counts[i] = input_size / world.size();
    if (i < input_size % world.size()) {
      recv_counts[i]++;
    }
    displs[i] = (i > 0) ? (displs[i - 1] + recv_counts[i - 1]) : 0;
  }

  boost::mpi::scatterv(world, input_vec, recv_counts, displs, local_input_vec.data(), 0);

  return true;
}

bool SentenceCountParallel::ValidationImpl() {
  if (world.rank() == 0) {
    return task_data->inputs_count[0] == input_size && task_data->outputs_count[0] == 1;
  }
  return true;
}

bool SentenceCountParallel::RunImpl() {
  local_sentence_count = 0;

  for (int i = 0; i < portion_size; ++i) {
    if (ispunct(local_input_vec[i]) &&
        (local_input_vec[i] == '.' || local_input_vec[i] == '!' || local_input_vec[i] == '?')) {
      if (i == portion_size - 1 || isspace(local_input_vec[i + 1])) {
        local_sentence_count++;
      }
    }
  }

  return true;
}

bool SentenceCountParallel::PostProcessingImpl() {
  int total_sentence_count = 0;

  boost::mpi::reduce(world, local_sentence_count, total_sentence_count, std::plus<int>(), 0);

  if (world.rank() == 0) {
    reinterpret_cast<int *>(task_data->outputs[0])[0] = total_sentence_count;
  }
}

bool fomin_v_sentence_count::SentenceCountSequential::PreProcessingImpl() {
  input_ = reinterpret_cast<char *>(task_data->inputs[0]);
  sentence_count = 0;
  return true;
}

bool fomin_v_sentence_count::SentenceCountSequential::ValidationImpl() {
  return task_data->inputs_count[0] == 1 && task_data->outputs_count[0] == 1;
}

bool fomin_v_sentence_count::SentenceCountSequential::RunImpl() {
  for (int i = 0; input_[i] != '\0'; ++i) {
    if (ispunct(input_[i]) && (input_[i] == '.' || input_[i] == '!' || input_[i] == '?')) {
      sentence_count++;
    }
  }
  return true;
}

bool fomin_v_sentence_count::SentenceCountSequential::PostProcessingImpl() {
  reinterpret_cast<int *>(task_data->outputs[0])[0] = sentence_count;
  return true;
}