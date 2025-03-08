
#include "mpi/vinyaikina_e_max_of_vector_elements/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/collectives/broadcast.hpp>
#include <boost/mpi/collectives/reduce.hpp>
#include <boost/mpi/collectives/scatterv.hpp>
#include <boost/mpi/operations.hpp>
#include <boost/serialization/vector.hpp>  //NOLINT
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

// Sequential Version
bool vinyaikina_e_max_of_vector_elements::VectorMaxSeq::ValidationImpl() {
  return !task_data->outputs.empty() && task_data->outputs_count[0] == 1;
}

bool vinyaikina_e_max_of_vector_elements::VectorMaxSeq::PreProcessingImpl() {
  auto* input_ptr = reinterpret_cast<int32_t*>(task_data->inputs[0]);
  input_.resize(task_data->inputs_count[0]);
  std::copy(input_ptr, input_ptr + task_data->inputs_count[0], input_.begin());
  return true;
}

bool vinyaikina_e_max_of_vector_elements::VectorMaxSeq::RunImpl() {
  if (input_.empty()) {
    return true;
  }
  max_ = -100000;
  for (int32_t num : input_) {
    max_ = std::max(num, max_);
    return true;
  }
  return true;
}

bool vinyaikina_e_max_of_vector_elements::VectorMaxSeq::PostProcessingImpl() {
  *reinterpret_cast<int32_t*>(task_data->outputs[0]) = max_;
  return true;
}

// Parallel Version
bool vinyaikina_e_max_of_vector_elements::VectorMaxPar::ValidationImpl() {
  return !task_data->outputs.empty() && task_data->outputs_count[0] == 1;
}

bool vinyaikina_e_max_of_vector_elements::VectorMaxPar::PreProcessingImpl() {
  max_ = std::numeric_limits<int32_t>::min();
  return true;
}

bool vinyaikina_e_max_of_vector_elements::VectorMaxPar::RunImpl() {
  int my_rank = world_.rank();
  int world_size = world_.size();
  int total_size = 0;

  if (my_rank == 0) {
    total_size = static_cast<int>(task_data->inputs_count[0]);
    auto* input_ptr = reinterpret_cast<int32_t*>(task_data->inputs[0]);
    input_.assign(input_ptr, input_ptr + total_size);
  }

  boost::mpi::broadcast(world_, total_size, 0);

  int local_size = (total_size / world_size) + (my_rank < (total_size % world_size) ? 1 : 0);
  std::vector<int> send_counts(world_size, total_size / world_size);
  std::vector<int> offsets(world_size, 0);

  for (int i = 0; i < total_size % world_size; ++i) {
    send_counts[i]++;
  }
  for (int i = 1; i < world_size; ++i) {
    offsets[i] = offsets[i - 1] + send_counts[i - 1];
  }

  local_input_.resize(send_counts[my_rank]);
  boost::mpi::scatterv(world_, input_.data(), send_counts, offsets, local_input_.data(), local_size, 0);

  int32_t local_max = std::numeric_limits<int32_t>::min();
  for (int32_t num : local_input_) {
    local_max = std::max(num, local_max);
  }
  boost::mpi::reduce(world_, local_max, max_, boost::mpi::maximum<int32_t>(), 0);
  return true;
}

bool vinyaikina_e_max_of_vector_elements::VectorMaxPar::PostProcessingImpl() {
  if (world_.rank() == 0) {
    *reinterpret_cast<int32_t*>(task_data->outputs[0]) = max_;
  }
  return true;
}
