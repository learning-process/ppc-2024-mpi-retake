#include "mpi/konkov_i_task_dining_philosophers_mpi/include/ops_mpi.hpp"

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <iostream>
#include <vector>

#include "core/task/include/task.hpp"

namespace konkov_i_task_dining_philosophers_mpi {

bool DiningPhilosophersMPI::PreProcessingImpl() {
  num_philosophers_ = static_cast<int>(task_data->inputs_count[0]);
  return num_philosophers_ > 1;
}

bool DiningPhilosophersMPI::ValidationImpl() { return num_philosophers_ > 1; }

bool DiningPhilosophersMPI::RunImpl() {
  int rank = world_.rank();
  int size = world_.size();

  for (int i = 0; i < num_philosophers_; ++i) {
    if (i % size == rank) {
      int left_fork = i;
      int right_fork = (i + 1) % num_philosophers_;

      if (i % 2 == 0) {
        std::cout << "Philosopher " << i << " (Process " << rank << ") waits for left fork " << left_fork << ".\n";
        boost::mpi::broadcast(world_, left_fork, rank);

        std::cout << "Philosopher " << i << " (Process " << rank << ") waits for right fork " << right_fork << ".\n";
        boost::mpi::broadcast(world_, right_fork, rank);
      } else {
        std::cout << "Philosopher " << i << " (Process " << rank << ") waits for right fork " << right_fork << ".\n";
        boost::mpi::broadcast(world_, right_fork, rank);

        std::cout << "Philosopher " << i << " (Process " << rank << ") waits for left fork " << left_fork << ".\n";
        boost::mpi::broadcast(world_, left_fork, rank);
      }

      std::cout << "Philosopher " << i << " (Process " << rank << ") is eating.\n";

      std::cout << "Philosopher " << i << " (Process " << rank << ") releases forks.\n";
    }

    world_.barrier();
  }

  return true;
}

bool DiningPhilosophersMPI::PostProcessingImpl() { return true; }

}  // namespace konkov_i_task_dining_philosophers_mpi
