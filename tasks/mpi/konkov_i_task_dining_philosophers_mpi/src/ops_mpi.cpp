#include "mpi/konkov_i_task_dining_philosophers_mpi/include/ops_mpi.hpp"

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <vector>

#include "boost/mpi/collectives/broadcast.hpp"

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

  for (int i = 0; i < num_philosophers_; ++i) {
    int left_fork = i;
    int right_fork = (i + 1) % num_philosophers_;

    boost::mpi::broadcast(world_, left_fork, i % size);
    boost::mpi::broadcast(world_, right_fork, i % size);
  }

  return true;
}

bool DiningPhilosophersMPI::PostProcessingImpl() { return true; }

}  // namespace konkov_i_task_dining_philosophers_mpi
