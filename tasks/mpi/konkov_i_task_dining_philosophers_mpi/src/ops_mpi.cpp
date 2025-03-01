#include "mpi/konkov_i_task_dining_philosophers_mpi/include/ops_mpi.hpp"

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <iostream>
#include <vector>

#include "core/task/include/task.hpp"

namespace konkov_i_task_dining_philosophers_mpi {

bool DiningPhilosophersMPI::PreProcessingImpl() {
  if (world_.rank() == 0) {
    std::cout << "[DEBUG] inputs_count size: " << task_data->inputs_count.size() << std::endl;
    if (!task_data->inputs_count.empty()) {
      std::cout << "[DEBUG] First input value: " << task_data->inputs_count[0] << std::endl;
    }
  }

  
  // Рассылаем значение всем процессам
  boost::mpi::broadcast(world_, num_philosophers_, 0);

  for (int r = 0; r < world_.size(); ++r) {
    world_.barrier();
    if (world_.rank() == r) {
      std::cout << "[Rank " << r << "] PreProcessing: philosophers = " << num_philosophers_ << std::endl;
    }
  }

  return num_philosophers_ > 1;
}

// В ValidationImpl
bool DiningPhilosophersMPI::ValidationImpl() {
  for (int r = 0; r < world_.size(); ++r) {
    world_.barrier();
    if (world_.rank() == r) {
      std::cout << "[Rank " << r << "] Validation: philosophers = " << num_philosophers_ << std::endl;
    }
  }
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
