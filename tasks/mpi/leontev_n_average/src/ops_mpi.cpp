// Copyright 2023 Nesterov Alexander
#include "mpi/leontev_n_average/include/ops_mpi.hpp"

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/collectives/broadcast.hpp>
#include <boost/mpi/collectives/reduce.hpp>
#include <boost/mpi/communicator.hpp>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <numeric>
#include <vector>

bool leontev_n_average_mpi::MPIVecAvgParallel::PreProcessingImpl() { return true; }

bool leontev_n_average_mpi::MPIVecAvgParallel::ValidationImpl() {
  if (world_.rank() == 0) {
    // Check count elements of output and 0 size
    return task_data->inputs_count[0] != 0 && task_data->outputs_count[0] == 1;
  }
  return true;
}

bool leontev_n_average_mpi::MPIVecAvgParallel::RunImpl() {
  std::div_t divres;

  if (world_.rank() == 0) {
    divres = std::div(static_cast<int>(task_data->inputs_count[0]), world_.size());
  }

  broadcast(world_, divres.quot, 0);
  broadcast(world_, divres.rem, 0);

  if (world_.rank() == 0) {
    input_ = std::vector<int>(task_data->inputs_count[0]);
    int* vec_ptr = reinterpret_cast<int*>(task_data->inputs[0]);

    for (size_t i = 0; i < task_data->inputs_count[0]; i++) {
      input_[i] = vec_ptr[i];
    }

    for (int proc = 1; proc < world_.size(); proc++) {
      int send_size = (proc == world_.size() - 1) ? divres.quot + divres.rem : divres.quot;
      world_.send(proc, 0, input_.data() + (proc * divres.quot), send_size);
    }
  }
  local_input_ = std::vector<int>((world_.rank() == world_.size() - 1) ? divres.quot + divres.rem : divres.quot);
  if (world_.rank() == 0) {
    local_input_ = std::vector<int>(input_.begin(), input_.begin() + divres.quot);
  } else {
    int recv_size = (world_.rank() == world_.size() - 1) ? divres.quot + divres.rem : divres.quot;
    world_.recv(0, 0, local_input_.data(), recv_size);
  }
  int local_res = std::accumulate(local_input_.begin(), local_input_.end(), 0);
  reduce(world_, local_res, res_, std::plus(), 0);
  if (world_.rank() == 0) {
    res_ = res_ / static_cast<int>(input_.size());
  }
  return true;
}

bool leontev_n_average_mpi::MPIVecAvgParallel::PostProcessingImpl() {
  if (world_.rank() == 0) {
    reinterpret_cast<int*>(task_data->outputs[0])[0] = res_;
  }
  return true;
}
