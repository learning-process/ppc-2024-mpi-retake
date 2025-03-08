#include "mpi/konkov_i_task_dining_philosophers_mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <random>
#include <vector>

#include "boost/mpi/collectives/all_gather.hpp"

bool konkov_i_task_dining_philosophers_mpi::DiningPhilosophersMPI::PreProcessingImpl() {
  l_philosopher_ = (world_.rank() + world_.size() - 1) % world_.size();
  r_philosopher_ = (world_.rank() + 1) % world_.size();
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> distrib(0, 2);
  status_ = distrib(gen);
  return true;
}

bool konkov_i_task_dining_philosophers_mpi::DiningPhilosophersMPI::ValidationImpl() {
  if (world_.rank() == 0) {
    count_philosophers_ = static_cast<int>(task_data->inputs_count[0]);
  } else {
    count_philosophers_ = world_.size();
  }

  return count_philosophers_ >= 2;
}

void konkov_i_task_dining_philosophers_mpi::DiningPhilosophersMPI::ReleaseForks() {
  status_ = 0;

  if (world_.iprobe(l_philosopher_, 0)) {
    world_.send(l_philosopher_, 0, status_);
    int recv_status = -2;
    world_.recv(l_philosopher_, 0, recv_status);
  }
  if (world_.iprobe(r_philosopher_, 0)) {
    world_.send(r_philosopher_, 0, status_);
    int recv_status = -2;
    world_.recv(r_philosopher_, 0, recv_status);
  }
}

bool konkov_i_task_dining_philosophers_mpi::DiningPhilosophersMPI::DistributionForks() {
  status_ = 2;
  int l_status = -1;
  int r_status = -1;

  auto request_fork = [&](int sosed, int& status) {
    world_.isend(sosed, 0, status_);
    if (world_.iprobe(sosed, 0)) {
      world_.irecv(sosed, 0, status);
      return (status == 0);
    }
    return false;
  };

  if (world_.rank() % 2 == 0) {
    if (request_fork(l_philosopher_, l_status) && request_fork(r_philosopher_, r_status)) {
      status_ = 1;
      world_.isend(l_philosopher_, 0, status_);
      world_.isend(r_philosopher_, 0, status_);
    }
  } else {
    if (request_fork(r_philosopher_, r_status) && request_fork(l_philosopher_, l_status)) {
      status_ = 1;
      world_.isend(l_philosopher_, 0, status_);
      world_.isend(r_philosopher_, 0, status_);
    }
  }
  return true;
}

bool konkov_i_task_dining_philosophers_mpi::DiningPhilosophersMPI::RunImpl() {
  do {
    status_ = 0;
    DistributionForks();
    status_ = 1;
    ReleaseForks();
    if (CheckDeadlock()) {
      return false;
    }
  } while (!CheckAllThink());
  return true;
}

bool konkov_i_task_dining_philosophers_mpi::DiningPhilosophersMPI::CheckAllThink() {
  std::vector<int> all_states;
  boost::mpi::all_gather(world_, status_, all_states);
  world_.barrier();
  return std::ranges::all_of(all_states, [](int state) { return state == 0; });
}

bool konkov_i_task_dining_philosophers_mpi::DiningPhilosophersMPI::CheckDeadlock() {
  std::vector<int> all_states(world_.size(), 0);
  boost::mpi::all_gather(world_, status_, all_states);
  return std::ranges::all_of(all_states, [](int state) { return state == 2; });
}

bool konkov_i_task_dining_philosophers_mpi::DiningPhilosophersMPI::PostProcessingImpl() {
  world_.barrier();
  int max_attempts = 100;
  int attempts = 0;
  while (world_.iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG) && attempts < max_attempts) {
    int last_message = 0;
    world_.recv(MPI_ANY_SOURCE, MPI_ANY_TAG, last_message);
    attempts++;
  }

  world_.barrier();
  return true;
}