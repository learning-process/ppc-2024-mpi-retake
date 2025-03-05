#include "mpi/sedova_o_linear_topology/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/mpi/collectives/all_gather.hpp>
#include <boost/mpi/collectives/broadcast.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/serialization/serialization.hpp>
#include <cmath>
#include <cstddef>
#include <random>
#include <vector>

std::vector<int> sedova_o_linear_topology_mpi::GetRandomVector(size_t size) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_int_distribution<> distrib(1, 500);
  std::vector<int> vec(size);
  for (size_t i = 0; i < size; i++) {
    vec[i] = distrib(gen);
  }
  return vec;
}

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
  if (world_.rank() == 0) {
    input_size = task_data->inputs_count[0];
  }
  boost::mpi::broadcast(world_, input_size, 0);
  input_.resize(input_size);
  if (world_.rank() == 0) {
    int *tmp_ptr = reinterpret_cast<int *>(task_data->inputs[0]);
    std::copy(tmp_ptr, tmp_ptr + input_size, input_.begin());
  }
  boost::mpi::broadcast(world_, input_, 0);
  std::vector<int> all_ranks(world_.size());
  boost::mpi::all_gather(world_, world_.rank(), all_ranks);
  bool order_is_ok = true;
  for (size_t i = 0; i < all_ranks.size(); ++i) {
    if (all_ranks[i] != static_cast<int>(i)) {
      order_is_ok = false;
      break;
    }
  }
  rec_ = order_is_ok;
  boost::mpi::broadcast(world_, rec_, 0);
  return true;
}

bool sedova_o_linear_topology_mpi::TestTaskMPI::PostProcessingImpl() {
  if (world_.rank() == 0) {
    bool *output = reinterpret_cast<bool *>(task_data->outputs[0]);
    output[0] = world_.size() == 1 ? true : rec_;
  }
  return true;
}
