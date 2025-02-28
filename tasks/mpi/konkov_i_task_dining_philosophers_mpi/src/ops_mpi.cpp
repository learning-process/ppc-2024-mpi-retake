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
    if (i % size == rank) {  // Каждый процесс обрабатывает философов циклически
      std::cout << "Process " << rank << " manages Philosopher " << i << ": thinking.\n";
      boost::mpi::broadcast(world_, i, rank);
      std::cout << "Process " << rank << " manages Philosopher " << i << ": eating.\n";
    }
    world_.barrier();
  }

  return true;
}

bool DiningPhilosophersMPI::PostProcessingImpl() { return true; }

}  // namespace konkov_i_task_dining_philosophers_mpi
