// Copyright 2023 Nesterov Alexander
#include "mpi/leontev_n_average/include/ops_mpi.hpp"

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/collectives/broadcast.hpp>
#include <boost/mpi/communicator.hpp>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <numeric>
#include <vector>

bool leontev_n_average_mpi::MPIVecAvgParallel::PreProcessingImpl() { return true; }

bool leontev_n_average_mpi::MPIVecAvgParallel::ValidationImpl() {
  if (world.rank() == 0) {
    // Check count elements of output and 0 size
    return task_data->inputs_count[0] != 0 && task_data->outputs_count[0] == 1;
  }
  return true;
}

bool leontev_n_average_mpi::MPIVecAvgParallel::RunImpl() {
  std::div_t divres;

  if (world.rank() == 0) {
    divres = std::div(task_data->inputs_count[0], world.size());
  }

  broadcast(world, divres.quot, 0);
  broadcast(world, divres.rem, 0);

  if (world.rank() == 0) {
    input_ = std::vector<int>(task_data->inputs_count[0]);
    int* vec_ptr = reinterpret_cast<int*>(task_data->inputs[0]);

    for (size_t i = 0; i < task_data->inputs_count[0]; i++) {
      input_[i] = vec_ptr[i];
    }

    for (int proc = 1; proc < world.size(); proc++) {
      int send_size = (proc == world.size() - 1) ? divres.quot + divres.rem : divres.quot;
      world.send(proc, 0, input_.data() + proc * divres.quot, send_size);
    }
  }
  local_input_ = std::vector<int>((world.rank() == world.size() - 1) ? divres.quot + divres.rem : divres.quot);
  if (world.rank() == 0) {
    local_input_ = std::vector<int>(input_.begin(), input_.begin() + divres.quot);
  } else {
    int recv_size = (world.rank() == world.size() - 1) ? divres.quot + divres.rem : divres.quot;
    world.recv(0, 0, local_input_.data(), recv_size);
  }
  int local_res = std::accumulate(local_input_.begin(), local_input_.end(), 0);
  reduce(world, local_res, res, std::plus(), 0);
  if (world.rank() == 0) {
    res = res / input_.size();
  }
  return true;
}

bool leontev_n_average_mpi::MPIVecAvgParallel::PostProcessingImpl() {
  if (world.rank() == 0) {
    reinterpret_cast<int*>(task_data->outputs[0])[0] = res;
  }
  return true;
}
