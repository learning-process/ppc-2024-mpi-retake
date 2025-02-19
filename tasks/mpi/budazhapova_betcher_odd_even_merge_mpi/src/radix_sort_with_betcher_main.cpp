
#include <algorithm>
#include <boost/mpi/communicator.hpp>
#include <cstddef>
#include <vector>

#include "mpi/budazhapova_betcher_odd_even_merge_mpi/include/radix_sort_with_betcher.h"

namespace budazhapova_betcher_odd_even_merge_mpi {
namespace {

void CountingSort(std::vector<int>& arr, int exp) {
  size_t n = arr.size();
  std::vector<int> output(n);
  std::vector<int> count(10, 0);

  for (size_t i = 0; i < n; i++) {
    int index = (arr[i] / exp) % 10;
    count[index]++;
  }
  for (int i = 1; i < 10; i++) {
    count[i] += count[i - 1];
  }
  for (size_t i = n - 1; i >= 0; i--) {
    int index = (arr[i] / exp) % 10;
    output[count[index] - 1] = arr[i];
    count[index]--;
  }
  for (size_t i = 0; i < n; i++) {
    arr[i] = output[i];
  }
}

void RadixSort(std::vector<int>& arr) {
  auto max_num_iter = std::ranges::max_element(arr);
  int max_num = *max_num_iter;
  for (int exp = 1; (max_num / exp > 0); exp *= 10) {
    CountingSort(arr, exp);
  }
}
void OddEvenMerge(std::vector<int>& local, std::vector<int>& received_data) {
  std::vector<int> merged(local.size() + received_data.size());
  std::ranges::copy(local, merged.begin());
  std::ranges::copy(received_data, merged.begin() + static_cast<long>(local.size()));
  budazhapova_betcher_odd_even_merge_mpi::RadixSort(merged);
  // local.assign(merged.begin(), merged.begin() + local.size());
  local.assign(merged.begin(), merged.begin() + static_cast<long>(local.size()));
  // received_data.assign(merged.begin() + local.size(), merged.end());
  received_data.assign(merged.begin() + static_cast<long>(local.size()), merged.end());
}
}  // namespace

}  // namespace budazhapova_betcher_odd_even_merge_mpi
bool budazhapova_betcher_odd_even_merge_mpi::MergeSequential::PreProcessingImpl() {
  res_ = std::vector<int>(reinterpret_cast<int*>(task_data->inputs[0]),
                          reinterpret_cast<int*>(task_data->inputs[0]) + task_data->inputs_count[0]);
  return true;
}

bool budazhapova_betcher_odd_even_merge_mpi::MergeSequential::ValidationImpl() {
  return task_data->inputs_count[0] > 0;
}

bool budazhapova_betcher_odd_even_merge_mpi::MergeSequential::RunImpl() {
  budazhapova_betcher_odd_even_merge_mpi::RadixSort(res_);
  return true;
}

bool budazhapova_betcher_odd_even_merge_mpi::MergeSequential::PostProcessingImpl() {
  int* output = reinterpret_cast<int*>(task_data->outputs[0]);
  for (size_t i = 0; i < res_.size(); i++) {
    output[i] = res_[i];
  }
  return true;
}

bool budazhapova_betcher_odd_even_merge_mpi::MergeParallel::PreProcessingImpl() {
  if (world_.rank() == 0) {
    res_ = std::vector<int>(reinterpret_cast<int*>(task_data->inputs[0]),
                            reinterpret_cast<int*>(task_data->inputs[0]) + task_data->inputs_count[0]);
  }
  return true;
}

bool budazhapova_betcher_odd_even_merge_mpi::MergeParallel::ValidationImpl() {
  if (world_.rank() == 0) {
    return task_data->inputs_count[0] > 0;
  }
  return true;
}

bool budazhapova_betcher_odd_even_merge_mpi::MergeParallel::RunImpl() {
std::vector<int> recv_counts(world_.size(), 0);
std::vector<int> displacements(world_.size(), 0);

boost::mpi::broadcast(world_, res_, 0);

int n_of_send_elements = 0;
int n_of_proc_with_extra_elements = 0;
int start = 0;
int end = 0;
int world_size = world_.size();
int world_rank = world_.rank();
int res_size = static_cast<int>(res_.size());

n_of_send_elements = res_size / world_size;
n_of_proc_with_extra_elements = res_size % world_size;

for (int i = 0; i < world_size; i++) {
  start = i * n_of_send_elements + std::min(i, n_of_proc_with_extra_elements);
  end = start + n_of_send_elements + (i < n_of_proc_with_extra_elements ? 1 : 0);
  recv_counts[i] = end - start;
  displacements[i] = (i == 0) ? 0 : displacements[i - 1] + recv_counts[i - 1];
}

start = world_rank * n_of_send_elements + std::min(world_rank, n_of_proc_with_extra_elements);
end = start + n_of_send_elements + (world_rank < n_of_proc_with_extra_elements ? 1 : 0);

local_res_.resize(end - start);
for (int i = start; i < end; i++) {
  local_res_[i - start] = res_[i];
}
for (int phase = 0; phase < world_size; ++phase) {
  bool is_even_phase = (phase % 2 == 0);
  int send_rank = is_even_phase ? (world_rank + 1) : (world_rank - 1);
  int recv_rank = is_even_phase ? (world_rank - 1) : (world_rank + 1);
  bool should_send = (is_even_phase && (world_rank % 2 == 0)) || (!is_even_phase && (world_rank % 2 == 1));

  if (should_send && (send_rank < world_size)) {
    world_.send(send_rank, world_rank, local_res_);
  }
  bool should_receive =
      (is_even_phase && (world_rank % 2 == 1)) || (!is_even_phase && (world_rank % 2 == 0) && (world_rank > 0));

  if (should_receive && (recv_rank < world_size)) {
    std::vector<int> received_data;
    world_.recv(recv_rank, recv_rank, received_data);
    OddEvenMerge(received_data, local_res_);
    world_.send(recv_rank, world_rank, received_data);
  }
  if (should_send && (send_rank < world_size)) {
    world_.recv(send_rank, send_rank, local_res_);
  }
}

boost::mpi::gatherv(world_, local_res_.data(), static_cast<int>(local_res_.size()), res_.data(), recv_counts,
                    displacements, 0);

return true;
}

bool budazhapova_betcher_odd_even_merge_mpi::MergeParallel::PostProcessingImpl() {
  if (world_.rank() == 0) {
    // budazhapova_betcher_odd_even_merge_mpi::RadixSort(res_);
    int* output = reinterpret_cast<int*>(task_data->outputs[0]);
    for (size_t i = 0; i < res_.size(); i++) {
      output[i] = res_[i];
    }
  }
  return true;
}
/* void SendAndReceive(boost::mpi::communicator& world_, int send_rank, int recv_rank, std::vector<int>& data) {
  if (send_rank >= 0 && send_rank < world_.size()) {
    world_.send(send_rank, world_.rank(), data);
  }
  if (recv_rank >= 0 && recv_rank < world_.size()) {
    world_.recv(recv_rank, recv_rank, data);
  }
}

void PerformOddEvenMerge(boost::mpi::communicator& world, int neighbor_rank, std::vector<int>& local_data) {
  std::vector<int> received_data;
  world_.recv(neighbor_rank, neighbor_rank, received_data);
  OddEvenMerge(&received_data, &local_data);
  world_.send(neighbor_rank, world_.rank(), received_data);
}

void OddEvenSortPhase(boost::mpi::communicator& world, int phase) {
  int next_rank = world_.rank() + 1;
  int prev_rank = world_.rank() - 1;

  if (phase % 2 == 0) {
    if (world_.rank() % 2 == 0 && next_rank < world_.size()) {
      SendAndReceive(next_rank, -1, local_res_);
    } else if (world_.rank() % 2 == 1) {
      PerformOddEvenMerge(prev_rank, local_res_);
    }
    if (world_.rank() % 2 == 0 && next_rank < world_.size()) {
      SendAndReceive(-1, next_rank, local_res_);
    }
  } else {
    if (world_.rank() % 2 == 1 && next_rank < world_.size()) {
      SendAndReceive(next_rank, -1, local_res_);
    } else if (world_.rank() % 2 == 0 && world_.rank() > 0) {
      PerformOddEvenMerge(prev_rank, local_res_);
    }
    if (world_.rank() % 2 == 1 && next_rank < world_.size()) {
      SendAndReceive(-1, next_rank, local_res_);
    }
  }
}

void DistributeData(boost::mpi::communicator& world, int& n_of_send_elements, int& n_of_proc_with_extra_elements,
                    int& start, int& end, std::vector<int>& recv_counts, std::vector<int>& displacements) {
  int world_size = world_.size();
  int world_rank = world_.rank();
  int res_size = static_cast<int>(res_.size());

  n_of_send_elements = res_size / world_size;
  n_of_proc_with_extra_elements = res_size % world_size;

  for (int i = 0; i < world_size; i++) {
    start = i * n_of_send_elements + std::min(i, n_of_proc_with_extra_elements);
    end = start + n_of_send_elements + (i < n_of_proc_with_extra_elements ? 1 : 0);
    recv_counts[i] = end - start;
    displacements[i] = (i == 0) ? 0 : displacements[i - 1] + recv_counts[i - 1];
  }

  start = world_rank * n_of_send_elements + std::min(world_rank, n_of_proc_with_extra_elements);
  end = start + n_of_send_elements + (world_rank < n_of_proc_with_extra_elements ? 1 : 0);

  local_res_.resize(end - start);
  for (int i = start; i < end; i++) {
    local_res_[i - start] = res_[i];
  }
}*/

/* bool budazhapova_betcher_odd_even_merge_mpi::MergeParallel::RunImpl() {
  std::vector<int> recv_counts(world_.size(), 0);
  std::vector<int> displacements(world_.size(), 0);

  boost::mpi::broadcast(world_, res_, 0);

  int n_of_send_elements = 0;
  int n_of_proc_with_extra_elements = 0;
  int start = 0;
  int end = 0;

  DistributeData(n_of_send_elements, n_of_proc_with_extra_elements, start, end, recv_counts, displacements);

  int world_size = world_.size();  // Переменная используется только здесь, поэтому выносим из DistributeData

  for (int phase = 0; phase < world_size; ++phase) {
    OddEvenSortPhase(phase);
  }

  boost::mpi::gatherv(world_, local_res_.data(), static_cast<int>(local_res_.size()), res_.data(), recv_counts,
                      displacements, 0);


  for (int phase = 0; phase < world_size; ++phase) {
    int next_rank = world_rank + 1;
    int prev_rank = world_rank - 1;

    if (phase % 2 == 0) {
      if (world_rank % 2 == 0 && next_rank < world_size) {
        world_.send(next_rank, world_rank, local_res_);
      } else if (world_rank % 2 == 1) {
        std::vector<int> received_data;
        world_.recv(prev_rank, prev_rank, received_data);
        OddEvenMerge(received_data, local_res_);
        world_.send(prev_rank, world_rank, received_data);
      }
      if (world_rank % 2 == 0 && next_rank < world_size) {
        world_.recv(next_rank, next_rank, local_res_);
      }
    } else {
      if (world_rank % 2 == 1 && next_rank < world_size) {
        world_.send(next_rank, world_rank, local_res_);
      } else if (world_rank % 2 == 0 && world_rank > 0) {
        std::vector<int> received_data;
        world_.recv(prev_rank, prev_rank, received_data);
        OddEvenMerge(received_data, local_res_);
        world_.send(prev_rank, world_rank, received_data);
      }
      if (world_rank % 2 == 1 && next_rank < world_size) {
        world_.recv(next_rank, next_rank, local_res_);
      }
    }
  }*/
