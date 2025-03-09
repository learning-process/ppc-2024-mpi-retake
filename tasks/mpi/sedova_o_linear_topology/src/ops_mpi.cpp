#include "mpi/sedova_o_linear_topology/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/mpi/collectives/all_gather.hpp>
#include <boost/mpi/collectives/broadcast.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/serialization/vector.hpp>  // NOLINT
#include <cmath>
#include <cstddef>
#include <functional>
#include <vector>

bool sedova_o_linear_topology_mpi::TestTaskMPI::PreProcessingImpl() {
  if (world_.size() == 1) {
    return true;
  }
  unsigned int input_size = 0;
  if (world_.rank() == 0) {
    output_.resize(0);
    input_size = task_data->inputs_count[0];
    int *tmp_ptr = reinterpret_cast<int *>(task_data->inputs[0]);
    input_.resize(input_size);
    std::copy(tmp_ptr, tmp_ptr + input_size, input_.begin());
  }
  return true;
}

bool sedova_o_linear_topology_mpi::TestTaskMPI::ValidationImpl() {
  return (world_.rank() != 0) || (task_data->inputs_count[0] > 0);
}

bool sedova_o_linear_topology_mpi::TestTaskMPI::RunImpl() {
  if (world_.size() == 1) {
    rec_ = true;
    return true;
  }
  unsigned int input_size = 0;
  std::vector<int> vec_;
  if (world_.rank() == 0) {
    input_size = task_data->inputs_count[0];
  }
  boost::mpi::broadcast(world_, input_size, 0);
  if (world_.rank() == world_.size() - 1) {
    vec_.resize(input_size);
  }
  input_.resize(input_size);
  if (world_.rank() == 0) {
    output_.push_back(0);
    world_.send(1, 0, input_);
    world_.send(1, 1, output_);
    world_.send(world_.size() - 1, 2, input_);
  } else {
    world_.recv(world_.rank() - 1, 0, input_);
    world_.recv(world_.rank() - 1, 1, output_);

    if (world_.rank() == world_.size() - 1) {
      world_.recv(0, 2, vec_);
    }

    output_.push_back(static_cast<size_t>(world_.rank()));

    if (world_.rank() != world_.size() - 1) {
      world_.send(world_.rank() + 1, 0, input_);
      world_.send(world_.rank() + 1, 1, output_);
    }
  }
  if (world_.rank() == world_.size() - 1) {
    bool output1_ = true;
    for (size_t i = 0; i < output_.size(); i++) {
      if (output_[i] != i) {
        output1_ = false;
        break;
      }
    }
    if (output_.size() != static_cast<size_t>(world_.size())) {
      output1_ = false;
    }

    if (input_ == vec_ && output1_) {
      world_.send(0, 3, true);
    } else {
      world_.send(0, 3, false);
    }
  }
  if (world_.rank() == 0) {
    world_.recv(world_.size() - 1, 3, rec_);
  }
  return true;
}
     
  

bool sedova_o_linear_topology_mpi::TestTaskMPI::PostProcessingImpl() {
  if (world_.rank() == 0) {
    bool *output = reinterpret_cast<bool *>(task_data->outputs[0]);
    if (world_.size() == 1) {
      output[0] = true;
    } else {
      output[0] = rec_;
    }
  }
  return true;
}
