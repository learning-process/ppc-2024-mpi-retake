#include <algorithm>
#include <boost/mpi/collectives/broadcast.hpp>
#include <boost/mpi/collectives/gather.hpp>
#include <boost/mpi/collectives/scatter.hpp>
#include <random>

#include "mpi/kalinin_d_odd_even_shellsort/include/header_mpi_odd_even_shell.hpp"

namespace kalinin_d_odd_even_shell_mpi {
void OddEvenShellMpi::ShellSort(std::vector<int>& vec) {
  int n = vec.size();
  for (int gap = n / 2; gap > 0; gap /= 2) {
    for (int i = gap; i < n; i++) {
      int temp = vec[i];
      int j;
      for (j = i; j >= gap && vec[j - gap] > temp; j -= gap) {
        vec[j] = vec[j - gap];
      }
      vec[j] = temp;
    }
  }
}
void GimmeRandVec(std::vector<int>& vec) {
  std::random_device rd;
  std::default_random_engine reng(rd());
  std::uniform_int_distribution<int> dist(0, static_cast<int>(vec.size()));
  std::generate(vec.begin(), vec.end(), [&dist, &reng] { return dist(reng); });
}

bool OddEvenShellMpi::PreProcessingImpl() {
  if (world_.rank() == 0) {
    int n = task_data->inputs_count[0];
    input_ = std::vector<int>(n);
    std::ranges::copy(reinterpret_cast<int*>(task_data->inputs[0]), reinterpret_cast<int*>(task_data->inputs[0]) + n,
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
    std::ranges::copy(output_, reinterpret_cast<int*>(task_data->outputs[0]));
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
    int reminder = (sz - (input_.size() % sz)) % sz;
    input_.resize(input_.size() + reminder, std::numeric_limits<int>::max());
    local_sz = input_.size() / sz;
  }
  broadcast(world_, local_sz, 0);

  local_vec.resize(local_sz);

  scatter(world_, input_, local_vec.data(), local_sz, 0);

  ShellSort(local_vec);
  int neighbour;
  for (int i = 0; i != sz; ++i) {
    int lower_bound = 0;
    int higher_bound = sz;
    if (i % 2 == 0) {
      higher_bound = is_even ? sz : sz - 1;

      if (id < lower_bound || id >= higher_bound) {
        continue;
      }

      neighbour = (id % 2 == 0) ? id + 1 : id - 1;
      if (neighbour < 0 || neighbour >= sz) {
        continue;
      }

      std::vector<int> received_data(local_sz);
      std::vector<int> merged(2 * local_sz);

      if (id % 2 == 0) {
        world_.send(neighbour, 0, local_vec);
        world_.recv(neighbour, 1, received_data);
      } else {
        world_.recv(neighbour, 0, received_data);
        world_.send(neighbour, 1, local_vec);
      }

      std::merge(local_vec.begin(), local_vec.end(), received_data.begin(), received_data.end(), merged.begin());

      if (id % 2 == 0) {
        local_vec.assign(merged.begin(), merged.begin() + local_sz);
      } else {
        local_vec.assign(merged.begin() + local_sz, merged.end());
      }

    } else {
      lower_bound = 1;
      higher_bound = is_even ? sz - 1 : sz;

      if (id < lower_bound || id >= higher_bound) {
        continue;
      }

      neighbour = (id % 2 != 0) ? id + 1 : id - 1;
      if (neighbour < 0 || neighbour >= sz) {
        continue;
      }

      std::vector<int> received_data(local_sz);
      std::vector<int> merged(2 * local_sz);

      if (id % 2 != 0) {
        world_.send(neighbour, 0, local_vec);
        world_.recv(neighbour, 1, received_data);
      } else {
        world_.recv(neighbour, 0, received_data);
        world_.send(neighbour, 1, local_vec);
      }

      std::merge(local_vec.begin(), local_vec.end(), received_data.begin(), received_data.end(), merged.begin());

      if (id % 2 != 0) {
        local_vec.assign(merged.begin(), merged.begin() + local_sz);
      } else {
        local_vec.assign(merged.begin() + local_sz, merged.end());
      }
    }
  }

  if (id != 0) {
    gather(world_, local_vec.data(), local_sz, 0);
  } else {
    output_.resize(input_.size());
    gather(world_, local_vec.data(), local_sz, output_, 0);
  }
  return true;
}

}  // namespace kalinin_d_odd_even_shell_mpi