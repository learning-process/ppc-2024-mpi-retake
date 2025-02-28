#include "mpi/konkov_i_task_dining_philosophers_mpi/include/ops_mpi.hpp"

#include <boost/mpi.hpp>
#include <boost/serialization/vector.hpp>
#include <chrono>
#include <iostream>
#include <thread>

namespace konkov_i_task_dp {

DiningPhilosophersMPI::DiningPhilosophersMPI(int num_philosophers) : num_philosophers_(num_philosophers) {
  std::cout << "Process " << world_.rank() << ": num_philosophers = " << num_philosophers_ << std::endl;
}

void DiningPhilosophersMPI::PreProcessing() {
  if (world_.rank() == 0) {
    std::cout << "Initializing dining philosophers problem with " << num_philosophers_ << " philosophers.\n";
  }
}

bool DiningPhilosophersMPI::Validation() { return num_philosophers_ > 1 && num_philosophers_ <= (world_.size() - 1); }

void DiningPhilosophersMPI::Run() {
  if (world_.rank() == 0) {
    ForkManagementProcess();
  } else {
    PhilosopherProcess(world_.rank() - 1);
  }
}

void DiningPhilosophersMPI::PostProcessing() {
  if (world_.rank() == 0) {
    std::cout << "Simulation completed successfully.\n";
  }
}

void DiningPhilosophersMPI::PhilosopherProcess(int id) {
  const int MAX_ITERATIONS = 3;
  boost::mpi::request reqs[2];

  for (int i = 0; i < MAX_ITERATIONS; ++i) {
    int left_fork = id;
    int right_fork = (id + 1) % num_philosophers_;

    reqs[0] = world_.isend(0, left_fork, id);
    reqs[1] = world_.isend(0, right_fork, id);
    boost::mpi::wait_all(reqs, reqs + 2);

    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    world_.send(0, left_fork, -1);
    world_.send(0, right_fork, -1);
  }

  world_.send(0, 0, -2);
}

void DiningPhilosophersMPI::ForkManagementProcess() {
  std::vector<std::queue<int>> fork_queues(num_philosophers_);
  int active_philosophers = num_philosophers_;

  while (active_philosophers > 0) {
    boost::mpi::status status = world_.probe();
    int philosopher_id;
    world_.recv(status.source(), status.tag(), philosopher_id);

    if (philosopher_id == -2) {
      active_philosophers--;
      continue;
    }

    int fork_id = status.tag();

    if (philosopher_id == -1) {
      if (!fork_queues[fork_id].empty()) {
        fork_queues[fork_id].pop();
        if (!fork_queues[fork_id].empty()) {
          int next_philosopher = fork_queues[fork_id].front();
          world_.send(next_philosopher, fork_id, 1);
        }
      }
    } else {
      fork_queues[fork_id].push(philosopher_id);
      if (fork_queues[fork_id].size() == 1) {
        world_.send(philosopher_id, fork_id, 1);
      }
    }
  }
}

}  // namespace konkov_i_task_dp