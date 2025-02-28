
#include "mpi/vinyaikina_e_max_of_vector_elements/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/mpi.hpp>
#include <boost/mpi/collectives/broadcast.hpp>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

namespace vinyaikina_e_max_of_vector_elements {

std::vector<int32_t> MakeRandomVector(int32_t size, int32_t val_min, int32_t val_max) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> distrib(val_min, val_max);

  std::vector<int32_t> new_vector(size);
  std::ranges::generate(new_vector.begin(), new_vector.end(), [&]() { return distrib(gen); });
  return new_vector;
}

// Sequential Version
bool VectorMaxSeq::ValidationImpl() { return !task_data->outputs.empty() && task_data->outputs_count[0] == 1; }

bool VectorMaxSeq::PreProcessingImpl() {
  auto* input_ptr = reinterpret_cast<int32_t*>(task_data->inputs[0]);
  input_.resize(task_data->inputs_count[0]);
  std::copy(input_ptr, input_ptr + task_data->inputs_count[0], input_.begin());
  return true;
}

bool VectorMaxSeq::RunImpl() {
  if (input_.empty()) {
    return true;
  }
  max_ = input_[0];
  for (int32_t num : input_) {
    max_ = std::max(num, max_);
    return true;
  }

  bool VectorMaxSeq::PostProcessingImpl() {
    *reinterpret_cast<int32_t*>(task_data->outputs[0]) = max_;
    return true;
  }

  // Parallel Version
  bool VectorMaxPar::ValidationImpl() { return !task_data->outputs.empty() && task_data->outputs_count[0] == 1; }

  bool VectorMaxPar::PreProcessingImpl() {
    max_ = std::numeric_limits<int32_t>::min();
    return true;
  }

  bool VectorMaxPar::RunImpl() {
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
      if (num > local_max) {
        local_max = num;
      }
    }
    boost::mpi::reduce(world_, local_max, max_, fmaximum<int32_t>(), 0);

    return true;
  }

  bool VectorMaxPar::PostProcessingImpl() {
    if (world_.rank() == 0) {
      *reinterpret_cast<int32_t*>(task_data->outputs[0]) = max_;
    }
    return true;
  }

}  // namespace vinyaikina_e_max_of_vector_elements
