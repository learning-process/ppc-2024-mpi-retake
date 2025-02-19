
#include <algorithm>
#include <boost/mpi/collectives/broadcast.hpp>
#include <boost/mpi/collectives/gatherv.hpp>
#include <boost/mpi/communicator.hpp>
#include <cstddef>
#include <numeric>
#include <vector>

#include "mpi/budazhapova_betcher_odd_even_merge_mpi/include/radix_sort_with_betcher.h"

namespace budazhapova_betcher_odd_even_merge_mpi {
namespace {
void CountingSort(std::vector<int>& arr, int exp) {
  size_t n = arr.size();
  std::vector<int> output(n);
  std::vector<int> count(10, 0);

  for (int i = 0; i < n; i++) {
    int index = (arr[i] / exp) % 10;
    count[index]++;
  }
  for (int i = 1; i < 10; i++) {
    count[i] += count[i - 1];
  }
  for (int i = n - 1; i >= 0; i--) {
    int index = (arr[i] / exp) % 10;
    output[count[index] - 1] = arr[i];
    count[index]--;
  }
  for (int i = 0; i < n; i++) {
    arr[i] = output[i];
  }
}

void RadixSort(std::vector<int>& arr) {
  int max_num = *std::max_element(arr.begin(), arr.end());
  for (int exp = 1; max_num / exp > 0; exp *= 10) {
    CountingSort(arr, exp);
  }
}
void OddEvenMerge(std::vector<int>& local_, std::vector<int>& received_data) {
  std::vector<int> merged(local_.size() + received_data.size());
  std::copy(local_.begin(), local_.end(), merged.begin());
  std::copy(received_data.begin(), received_data.end(), merged.begin() + local_.size());
  budazhapova_betcher_odd_even_merge_mpi::RadixSort(merged);
  local_.assign(merged.begin(), merged.begin() + local_.size());
  received_data.assign(merged.begin() + local_.size(), merged.end());
}
}  // namespace

}  // namespace budazhapova_betcher_odd_even_merge_mpi
bool budazhapova_betcher_oddEvenMerge_mpi::MergeSequential::PreProcessingImpl(){
  res_ = std::vector<int>(reinterpret_cast<int*>(task_data->inputs[0]),
                          reinterpret_cast<int*>(task_data->inputs[0]) + task_data->inputs_count[0]);
  return true;
}

bool budazhapova_betcher_odd_even_merge_mpi::MergeSequential::ValidationImpl(){
  return task_data->inputs_count[0] > 0;
}

bool budazhapova_betcher_odd_even_merge_mpi::MergeSequential::Run() {
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
  }

  boost::mpi::gatherv(world_, local_res_.data(), local_res_.size(), res_.data(), recv_counts, displacements, 0);

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