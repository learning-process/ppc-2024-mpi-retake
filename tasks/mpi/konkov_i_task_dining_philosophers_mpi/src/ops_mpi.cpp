#include "mpi/konkov_i_task_dining_philosophers_mpi/include/ops_mpi.hpp"

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <chrono>
#include <thread>
#include <vector>

namespace konkov_i_task_dining_philosophers_mpi {

bool DiningPhilosophersMPI::PreProcessingImpl() {
  if (world_.rank() == 0 && !task_data->inputs_count.empty()) {
    num_philosophers_ = static_cast<int>(task_data->inputs_count[0]);
  }

  boost::mpi::broadcast(world_, num_philosophers_, 0);
  return num_philosophers_ > 1;
}

bool DiningPhilosophersMPI::ValidationImpl() { return num_philosophers_ > 1; }

bool DiningPhilosophersMPI::RunImpl() {
  int rank = world_.rank();
  int size = world_.size();

  if (rank >= num_philosophers_) {
    return true;
  }

  int philosophers_per_process = num_philosophers_ / size;
  int remaining = num_philosophers_ % size;
  int start = rank * philosophers_per_process + std::min(rank, remaining);
  int end = start + philosophers_per_process + (rank < remaining ? 1 : 0);

  for (int i = start; i < end; ++i) {
    int left_fork = i;
    int right_fork = (i + 1) % num_philosophers_;
    // try
    if (i % 2 == 0) {
      boost::mpi::broadcast(world_, left_fork, rank);
      boost::mpi::broadcast(world_, right_fork, rank);
    } else {
      boost::mpi::broadcast(world_, right_fork, rank);
      boost::mpi::broadcast(world_, left_fork, rank);
    }
  }

  return true;
}

bool DiningPhilosophersMPI::PostProcessingImpl() { return true; }

}  // namespace konkov_i_task_dining_philosophers_mpi
