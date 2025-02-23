#include "mpi/konkov_i_task_dining_philosophers/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <vector>

namespace konkov_i_dining_philosophers {

DiningPhilosophers::DiningPhilosophers(int num_philosophers)
    : num_philosophers_(num_philosophers), fork_states_(num_philosophers, 0), philosopher_states_(num_philosophers, 0) {
  MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &size_);
}

bool DiningPhilosophers::Validation() const { return num_philosophers_ > 1; }

bool DiningPhilosophers::PreProcessing() {
  if (rank_ == 0) {
    InitPhilosophers();
  }
  MPI_Bcast(fork_states_.data(), num_philosophers_, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);
  return true;
}

bool DiningPhilosophers::Run() {
  for (int i = 0; i < num_philosophers_; ++i) {
    if (rank_ == i) {
      PhilosopherActions(i);
    }
    UpdateForkStates();
    MPI_Barrier(MPI_COMM_WORLD);
  }
  return true;
}

bool DiningPhilosophers::PostProcessing() { return !IsDeadlock(); }

int DiningPhilosophers::CheckDeadlock() { return IsDeadlock(); }

void DiningPhilosophers::InitPhilosophers() {
  std::ranges::fill(fork_states_, 0);
  std::ranges::fill(philosopher_states_, 0);
  ;
}

void DiningPhilosophers::PhilosopherActions(int id) {
  int left_fork = id;
  int right_fork = (id + 1) % num_philosophers_;

  if (fork_states_[left_fork] == 0 && fork_states_[right_fork] == 0) {
    fork_states_[left_fork] = 1;
    fork_states_[right_fork] = 1;
    philosopher_states_[id] = 1;  // Eating
  }

  fork_states_[left_fork] = 0;
  fork_states_[right_fork] = 0;
  philosopher_states_[id] = 0;  // Thinking
}

int DiningPhilosophers::IsDeadlock() {
  return std::ranges::all_of(philosopher_states_, [](int state) { return state == 1; }) ? 1 : 0;
}


void DiningPhilosophers::UpdateForkStates() {
  MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, fork_states_.data(), 1, MPI_INT, MPI_COMM_WORLD);
}

}  // namespace konkov_i_dining_philosophers