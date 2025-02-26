#include "mpi/konkov_i_task_dining_philosophers/include/ops_mpi.hpp"

#include <stdexcept>

DiningPhilosophersMPI::DiningPhilosophersMPI(int numPhilosophers) : numPhilosophers(numPhilosophers) {
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  comm = MPI_COMM_WORLD;
}

void DiningPhilosophersMPI::Validation() {
  if (numPhilosophers <= 0) {
    throw std::invalid_argument("Number of philosophers must be greater than zero.");
  }
}

void DiningPhilosophersMPI::PreProcessing() {
  int base = numPhilosophers / size;
  int remainder = numPhilosophers % size;

  localStart = rank * base + (rank < remainder ? rank : remainder);
  localEnd = localStart + base + (rank < remainder ? 1 : 0);
}

void DiningPhilosophersMPI::Run() {
  for (int i = 0; i < 5; ++i) {
    for (int id = localStart; id < localEnd; ++id) {
      Think(id);
      PickUpForks(id);
      Eat(id);
      PutDownForks(id);
    }
    MPI_Barrier(comm);
  }
}

void DiningPhilosophersMPI::PostProcessing() {
  int localSteps = (localEnd - localStart) * 5;
  int totalSteps = 0;
  MPI_Allreduce(&localSteps, &totalSteps, 1, MPI_INT, MPI_SUM, comm);
}

void DiningPhilosophersMPI::PickUpForks(int id) {
  int leftFork = id;
  int rightFork = (id + 1) % numPhilosophers;

  MPI_Send(nullptr, 0, MPI_INT, leftFork, 0, comm);
  MPI_Send(nullptr, 0, MPI_INT, rightFork, 0, comm);
}

void DiningPhilosophersMPI::PutDownForks(int id) {
  int leftFork = id;
  int rightFork = (id + 1) % numPhilosophers;

  MPI_Recv(nullptr, 0, MPI_INT, leftFork, 0, comm, MPI_STATUS_IGNORE);
  MPI_Recv(nullptr, 0, MPI_INT, rightFork, 0, comm, MPI_STATUS_IGNORE);
}

void DiningPhilosophersMPI::Think(int id) {}

void DiningPhilosophersMPI::Eat(int id) {}
