// Copyright 2023 Nesterov Alexander
#include "mpi/makhov_m_ring_topology/include/ops_mpi.hpp"

#include <algorithm>
#include <cstdint>
#include <vector>

bool makhov_m_ring_topology::TestMPITaskParallel::PreProcessingImpl() {
  // Init vector in root
  if (world_.rank() == 0) {
    sequence_.clear();
    input_data_ = std::vector<int32_t>(task_data->inputs_count[0]);
    auto* data_ptr = reinterpret_cast<int32_t*>(task_data->inputs[0]);
    std::copy(data_ptr, data_ptr + task_data->inputs_count[0], input_data_.begin());
  }
  return true;
}

bool makhov_m_ring_topology::TestMPITaskParallel::ValidationImpl() {
  if (world_.rank() == 0) {
    return task_data->inputs_count.size() == 1 && task_data->outputs_count.size() == 2 &&
           task_data->outputs_count[0] == task_data->inputs_count[0];
  }
  return true;
}

bool makhov_m_ring_topology::TestMPITaskParallel::RunImpl() {
  if (world_.size() < 2) {
    output_data_ = input_data_;
    sequence_.push_back(0);
  }

  else {
    if (world_.rank() == 0) {
      sequence_.push_back(world_.rank());
      world_.send(world_.rank() + 1, 0, input_data_);
      world_.send(world_.rank() + 1, 1, sequence_);

      int sender = world_.size() - 1;
      world_.recv(sender, 0, output_data_);
      world_.recv(sender, 1, sequence_);
      sequence_.push_back(world_.rank());
    } else {
      int sender = world_.rank() - 1;
      world_.recv(sender, 0, input_data_);
      world_.recv(sender, 1, sequence_);
      sequence_.push_back(world_.rank());

      int receiver = (world_.rank() + 1) % world_.size();
      world_.send(receiver, 0, input_data_);
      world_.send(receiver, 1, sequence_);
    }
  }
  return true;
}

bool makhov_m_ring_topology::TestMPITaskParallel::PostProcessingImpl() {
  if (world_.rank() == 0) {
    auto* output_ptr = reinterpret_cast<int32_t*>(task_data->outputs[0]);
    auto* sequence_ptr = reinterpret_cast<int32_t*>(task_data->outputs[1]);

    std::ranges::copy(input_data_, output_ptr);
    std::ranges::copy(sequence_, sequence_ptr);
  }
  return true;
}
