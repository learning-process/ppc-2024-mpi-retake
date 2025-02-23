// Copyright 2023 Nesterov Alexander
#include "mpi/makhov_m_ring_topology/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <string>

bool makhov_m_ring_topology::TestMPITaskParallel::PreProcessingImpl() {
  // internal_order_test();

  // Init vector in root
  if (world.rank() == 0) {
    sequence.clear();
    input_data = std::vector<int32_t>(task_data->inputs_count[0]);
    auto* data_ptr = reinterpret_cast<int32_t*>(task_data->inputs[0]);
    std::copy(data_ptr, data_ptr + task_data->inputs_count[0], input_data.begin());
  }
  return true;
}

bool makhov_m_ring_topology::TestMPITaskParallel::ValidationImpl() {
  //internal_order_test();
  if (world.rank() == 0) {
    return task_data->inputs_count.size() == 1 && task_data->inputs_count[0] >= 0 &&
           task_data->outputs_count.size() == 2 && task_data->outputs_count[0] == task_data->inputs_count[0];
  }
  return true;
}

bool makhov_m_ring_topology::TestMPITaskParallel::RunImpl() {
  //internal_order_test();
  if (world.size() < 2) {
    output_data = input_data;
    sequence.push_back(0);
  }

  else {
    if (world.rank() == 0) {
      sequence.push_back(world.rank());
      world.send(world.rank() + 1, 0, input_data);
      world.send(world.rank() + 1, 1, sequence);

      int sender = world.size() - 1;
      world.recv(sender, 0, output_data);
      world.recv(sender, 1, sequence);
      sequence.push_back(world.rank());
    } else {
      int sender = world.rank() - 1;
      world.recv(sender, 0, input_data);
      world.recv(sender, 1, sequence);
      sequence.push_back(world.rank());

      int receiver = (world.rank() + 1) % world.size();
      world.send(receiver, 0, input_data);
      world.send(receiver, 1, sequence);
    }
  }
  return true;
}

bool makhov_m_ring_topology::TestMPITaskParallel::PostProcessingImpl() {
  //internal_order_test();
  if (world.rank() == 0) {
    auto* output_ptr = reinterpret_cast<int32_t*>(task_data->outputs[0]);
    auto* sequence_ptr = reinterpret_cast<int32_t*>(task_data->outputs[1]);

    std::copy(input_data.begin(), input_data.end(), output_ptr);
    std::copy(sequence.begin(), sequence.end(), sequence_ptr);
  }
  return true;
}
