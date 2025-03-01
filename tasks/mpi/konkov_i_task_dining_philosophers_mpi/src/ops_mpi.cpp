#include "mpi/konkov_i_task_dining_philosophers_mpi/include/ops_mpi.hpp"

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <iostream>
#include <vector>

#include "core/task/include/task.hpp"

namespace konkov_i_task_dining_philosophers_mpi {

bool DiningPhilosophersMPI::PreProcessingImpl() {
  if (world_.rank() == 0 && !task_data->inputs_count.empty()) {
    num_philosophers_ = task_data->inputs_count[0];
  }

  world_.barrier();
  boost::mpi::broadcast(world_, num_philosophers_, 0);
  world_.barrier();

  return num_philosophers_ > 1;
}

bool DiningPhilosophersMPI::ValidationImpl() {
  world_.barrier();
  return num_philosophers_ > 1;
}

bool DiningPhilosophersMPI::RunImpl() {
  int rank = world_.rank();
  int size = world_.size();

  if (rank >= num_philosophers_) {
    world_.barrier();
    return true;
  }

  for (int i = 0; i < num_philosophers_; ++i) {
    int left_fork = i;
    int right_fork = (i + 1) % num_philosophers_;

    if (i % size == rank) {
      std::cout << "Philosopher " << i << " (Process " << rank << ") waits for left fork " << left_fork << ".\n";
    }
    boost::mpi::broadcast(world_, left_fork, i % size);

    if (i % size == rank) {
      std::cout << "Philosopher " << i << " (Process " << rank << ") waits for right fork " << right_fork << ".\n";
    }
    boost::mpi::broadcast(world_, right_fork, i % size);

    if (i % size == rank) {
      std::cout << "Philosopher " << i << " (Process " << rank << ") is eating.\n";
      std::cout << "Philosopher " << i << " (Process " << rank << ") releases forks.\n";
    }

    world_.barrier();
  }

  return true;
}

bool DiningPhilosophersMPI::PostProcessingImpl() { return true; }

}  // namespace konkov_i_task_dining_philosophers_mpi
