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

bool DiningPhilosophers::validation() const { return num_philosophers_ > 1; }

bool DiningPhilosophers::pre_processing() {
  if (rank_ == 0) {
    init_philosophers();
  }
  MPI_Bcast(fork_states_.data(), num_philosophers_, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);
  return true;
}

bool DiningPhilosophers::run() {
  for (int i = 0; i < num_philosophers_; ++i) {
    if (rank_ == i) {
      philosopher_actions(i);
    }
    update_fork_states();
    MPI_Barrier(MPI_COMM_WORLD);
  }
  return true;
}

bool DiningPhilosophers::post_processing() { return !is_deadlock(); }

bool DiningPhilosophers::check_deadlock() { return is_deadlock(); }

void DiningPhilosophers::init_philosophers() {
  std::fill(fork_states_.begin(), fork_states_.end(), 0);
  std::fill(philosopher_states_.begin(), philosopher_states_.end(), 0);
}

void DiningPhilosophers::philosopher_actions(int id) {
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

bool DiningPhilosophers::is_deadlock() {
  bool local_deadlock =
      std::all_of(philosopher_states_.begin(), philosopher_states_.end(), [](int state) { return state == 1; });
  bool global_deadlock = false;
  MPI_Allreduce(&local_deadlock, &global_deadlock, 1, MPI_C_BOOL, MPI_LOR, MPI_COMM_WORLD);
  return global_deadlock;
}

void DiningPhilosophers::update_fork_states() {
  MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, fork_states_.data(), 1, MPI_INT, MPI_COMM_WORLD);
}

}  // namespace konkov_i_dining_philosophers