#include <algorithm>
#include <boost/mpi/collectives/broadcast.hpp>
#include <boost/mpi/collectives/scatter.hpp>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <utility>
#include <vector>

#include "mpi/kalinin_d_odd_even_shellsort/include/header_mpi_odd_even_shell.hpp"

namespace kalinin_d_odd_even_shell_mpi {
void OddEvenShellMpi::ShellSort(std::vector<int> &vec) {
  int n = vec.empty() ? 0 : static_cast<int>(vec.size());
  for (int gap = n / 2; gap > 0; gap /= 2) {
    for (int i = gap; i < n; i++) {
      int temp = vec[i];
      int j = 0;
      for (j = i; j >= gap && vec[j - gap] > temp; j -= gap) {
        vec[j] = vec[j - gap];
      }
      vec[j] = temp;
    }
  }
}
void GimmeRandVec(std::vector<int> &vec) {
  std::random_device rd;
  std::default_random_engine reng(rd());
  std::uniform_int_distribution<int> dist(0, static_cast<int>(vec.size()));
  std::ranges::generate(vec, [&dist, &reng] { return dist(reng); });
}

bool OddEvenShellMpi::PreProcessingImpl() {
  if (world_.rank() == 0) {
    int n = static_cast<int>(task_data->inputs_count[0]);
    input_ = std::vector<int>(n);
    std::ranges::copy(reinterpret_cast<int *>(task_data->inputs[0]), reinterpret_cast<int *>(task_data->inputs[0]) + n,
                      input_.begin());
  }
  return true;
}

bool OddEvenShellMpi::ValidationImpl() {
  if (world_.rank() == 0) {
    return (task_data->outputs_count[0] > 0 && task_data->inputs_count[0] > 1);
  }
  return true;
}

bool OddEvenShellMpi::PostProcessingImpl() {
  if (world_.rank() == 0) {
    output_.resize(task_data->inputs_count[0]);
    std::ranges::copy(output_, reinterpret_cast<int *>(task_data->outputs[0]));
  }

  return true;
}
bool OddEvenShellMpi::RunImpl() {
  auto id = world_.rank();
  auto sz = world_.size();
  bool is_even = (sz % 2 == 0);
  std::vector<int> local_vec;
  int local_sz = 0;

  if (sz == 1) {
    output_ = std::move(input_);
    ShellSort(output_);
    return true;
  }

  if (id == 0) {
    PrepareInput(local_sz);
  }

  broadcast(world_, local_sz, 0);
  local_vec.resize(local_sz);
  scatter(world_, input_, local_vec.data(), local_sz, 0);
  ShellSort(local_vec);

  for (int i = 0; i != sz; ++i) {
    PerformOddEvenPhase(i, id, sz, is_even, local_vec, local_sz);
  }

  GatherResults(id, local_vec, local_sz);
  return true;
}

void OddEvenShellMpi::PrepareInput(int &local_sz) {
  size_t reminder = 0;
  reminder = (world_.size() - (input_.size() % world_.size())) % world_.size();
  input_.resize(input_.size() + reminder, std::numeric_limits<int>::max());
  local_sz = input_.size() / world_.size();
}

void OddEvenShellMpi::PerformOddEvenPhase(int phase, int id, int sz, bool is_even, std::vector<int> &local_vec,
                                          int local_sz) {
  int lower_bound = (phase % 2 == 0) ? 0 : 1;
  int higher_bound;
  if (phase % 2 == 0) {
    if (is_even) {
      higher_bound = sz;
    } else {
      higher_bound = sz - 1;
    }
  } else {
    if (is_even) {
      higher_bound = sz - 1;
    } else {
      higher_bound = sz;
    }
  }

  if (id < lower_bound || id >= higher_bound) {
    return;
  }

  int neighbour;
  if (phase % 2 == 0) {
    if (id % 2 == 0) {
      neighbour = id + 1;
    } else {
      neighbour = id - 1;
    }
  } else {
    if (id % 2 != 0) {
      neighbour = id + 1;
    } else {
      neighbour = id - 1;
    }
  }
  if (neighbour < 0 || neighbour >= sz) {
    return;
  }

  std::vector<int> received_data(local_sz);
  std::vector<int> merged(2 * local_sz);

  if (phase % 2 == 0) {
    ExchangeData(id, neighbour, local_vec, received_data, 0, 1);
  } else {
    ExchangeData(id, neighbour, local_vec, received_data, 1, 0);
  }

  std::ranges::merge(local_vec, received_data, merged.begin());
  if (phase % 2 == 0) {
    local_vec.assign(merged.begin(), merged.begin() + local_sz);
  } else {
    local_vec.assign(merged.begin() + local_sz, merged.end());
  }
}

void OddEvenShellMpi::ExchangeData(int id, int neighbour, std::vector<int> &local_vec, std::vector<int> &received_data,
                                   int send_tag, int recv_tag) {
  world_.send(neighbour, send_tag, local_vec);
  world_.recv(neighbour, recv_tag, received_data);
}

void OddEvenShellMpi::GatherResults(int id, std::vector<int> &local_vec, int local_sz) {
  if (id != 0) {
    gather(world_, local_vec.data(), local_sz, 0);
  } else {
    output_.resize(input_.size());
    gather(world_, local_vec.data(), local_sz, output_, 0);
  }
}

}  // namespace kalinin_d_odd_even_shell_mpi