#include "mpi/konkov_i_task_dining_philosophers/include/ops_mpi.hpp"

#include <chrono>
#include <iostream>
#include <thread>

using namespace dining_philosophers;

DiningPhilosophersMPI::DiningPhilosophersMPI(int num_philosophers) : num_philosophers(num_philosophers) {
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  states.resize(num_philosophers, 0);
}

DiningPhilosophersMPI::~DiningPhilosophersMPI() {}

void DiningPhilosophersMPI::Validation() {
  if (size < num_philosophers) {
    if (rank == 0) {
      std::cerr << "Error: Not enough processes for philosophers.\n";
    }
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
}

void DiningPhilosophersMPI::PreProcessing() {
  if (rank == 0) {
    std::cout << "Starting Dining Philosophers problem with " << num_philosophers << " philosophers.\n";
  }
}

void DiningPhilosophersMPI::Run() {
  for (int i = 0; i < 5; ++i) {
    Think(rank);
    TakeForks(rank);
    Eat(rank);
    PutForks(rank);
  }
}

void DiningPhilosophersMPI::PostProcessing() {
  if (rank == 0) {
    std::cout << "Dining Philosophers problem finished.\n";
  }
}

void DiningPhilosophersMPI::Think(int philosopher_id) { std::this_thread::sleep_for(std::chrono::milliseconds(100)); }

void DiningPhilosophersMPI::Eat(int philosopher_id) { std::this_thread::sleep_for(std::chrono::milliseconds(100)); }

void DiningPhilosophersMPI::TakeForks(int philosopher_id) {
  MPI_Send(nullptr, 0, MPI_INT, (philosopher_id + 1) % num_philosophers, 0, MPI_COMM_WORLD);
}

void DiningPhilosophersMPI::PutForks(int philosopher_id) {
  MPI_Recv(nullptr, 0, MPI_INT, (philosopher_id + 1) % num_philosophers, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}
