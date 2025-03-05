#define OMPI_SKIP_MPICXX
#include "mpi/muradov_k_radix_sort/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <functional>
#include <utility>
#include <vector>

namespace muradov_k_radix_sort {

namespace {
void CountingSortForRadix(std::vector<int>& arr, int exp) {
  int n = static_cast<int>(arr.size());
  std::vector<int> output(n);
  int count[10] = {0};
  for (int i = 0; i < n; ++i) {
    int digit = (arr[i] / exp) % 10;
    count[digit]++;
  }
  for (int i = 1; i < 10; ++i) {
    count[i] += count[i - 1];
  }
  for (int i = n - 1; i >= 0; --i) {
    int digit = (arr[i] / exp) % 10;
    output[count[digit] - 1] = arr[i];
    count[digit]--;
  }
  for (int i = 0; i < n; ++i) {
    arr[i] = output[i];
  }
}

void LSDRadixSort(std::vector<int>& arr) {
  if (arr.empty()) return;
  int max_val = *std::max_element(arr.begin(), arr.end());
  for (int exp = 1; max_val / exp > 0; exp *= 10) {
    CountingSortForRadix(arr, exp);
  }
}

void SequentialRadixSort(std::vector<int>& v) {
  std::vector<int> negatives;
  std::vector<int> non_negatives;
  for (int x : v) {
    if (x < 0)
      negatives.push_back(-x);
    else
      non_negatives.push_back(x);
  }
  LSDRadixSort(non_negatives);
  LSDRadixSort(negatives);
  std::reverse(negatives.begin(), negatives.end());
  for (int& x : negatives) {
    x = -x;
  }
  int index = 0;
  for (int x : negatives) {
    v[index++] = x;
  }
  for (int x : non_negatives) {
    v[index++] = x;
  }
}

void MergeAscending(std::vector<int>& local_part, const std::vector<int>& neighbor_part, std::vector<int>& tmp) {
  std::size_t i = 0, j = 0, k = 0;
  while (i < local_part.size() && j < neighbor_part.size()) {
    if (local_part[i] < neighbor_part[j])
      tmp[k++] = local_part[i++];
    else
      tmp[k++] = neighbor_part[j++];
  }
  while (i < local_part.size()) tmp[k++] = local_part[i++];
  while (j < neighbor_part.size()) tmp[k++] = neighbor_part[j++];
  local_part = tmp;
}

void MergeDescending(std::vector<int>& local_part, const std::vector<int>& neighbor_part, std::vector<int>& tmp) {
  int i = static_cast<int>(local_part.size()) - 1;
  int j = static_cast<int>(neighbor_part.size()) - 1;
  int k = static_cast<int>(local_part.size()) - 1;
  while (i >= 0 && j >= 0) {
    if (local_part[i] > neighbor_part[j])
      tmp[k--] = local_part[i--];
    else
      tmp[k--] = neighbor_part[j--];
  }
  while (i >= 0) tmp[k--] = local_part[i--];
  while (j >= 0) tmp[k--] = neighbor_part[j--];
  local_part = tmp;
}

std::vector<std::pair<int, int>> BuildAllocation(int proc_num) {
  std::vector<int> v(proc_num);
  for (int i = 0; i < proc_num; ++i) {
    v[i] = i;
  }
  std::vector<std::pair<int, int>> schedule;
  std::function<void(const std::vector<int>&)> Allocation = [&](const std::vector<int>& vec) {
    int size = static_cast<int>(vec.size());
    if (size <= 1) return;
    int mid = size / 2;
    std::vector<int> left(vec.begin(), vec.begin() + mid);
    std::vector<int> right(vec.begin() + mid, vec.end());
    Allocation(left);
    Allocation(right);
    for (std::size_t i = 0; i < left.size() && i < right.size(); ++i) {
      schedule.emplace_back(left[i], right[i]);
    }
  };
  Allocation(v);
  return schedule;
}
}  // namespace

void MPI_RadixSort(std::vector<int>& v) {
  int proc_rank = 0, proc_count = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &proc_count);
  if (proc_count <= 1 || static_cast<int>(v.size()) <= proc_count) {
    if (proc_rank == 0 && !v.empty()) {
      SequentialRadixSort(v);
    }
    return;
  }
  std::vector<std::pair<int, int>> schedule = BuildAllocation(proc_count);
  int padding_count = 0;
  if (proc_rank == 0) {
    while (v.size() % proc_count != 0) {
      v.push_back(0);
      ++padding_count;
    }
  }
  int enlarged_size = 0;
  if (proc_rank == 0) {
    enlarged_size = static_cast<int>(v.size());
  }
  MPI_Bcast(&enlarged_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
  int part_size = enlarged_size / proc_count;
  std::vector<int> local_part(part_size);
  MPI_Scatter(v.data(), part_size, MPI_INT, local_part.data(), part_size, MPI_INT, 0, MPI_COMM_WORLD);
  SequentialRadixSort(local_part);
  std::vector<int> tmp(part_size);
  std::vector<int> neighbor_part(part_size);
  for (std::size_t i = 0; i < schedule.size(); ++i) {
    int proc_first = schedule[i].first;
    int proc_second = schedule[i].second;
    MPI_Status status;
    if (proc_rank == proc_second) {
      MPI_Recv(neighbor_part.data(), part_size, MPI_INT, proc_first, 0, MPI_COMM_WORLD, &status);
      MPI_Send(local_part.data(), part_size, MPI_INT, proc_first, 0, MPI_COMM_WORLD);
      MergeDescending(local_part, neighbor_part, tmp);
    } else if (proc_rank == proc_first) {
      MPI_Send(local_part.data(), part_size, MPI_INT, proc_second, 0, MPI_COMM_WORLD);
      MPI_Recv(neighbor_part.data(), part_size, MPI_INT, proc_second, 0, MPI_COMM_WORLD, &status);
      MergeAscending(local_part, neighbor_part, tmp);
    }
  }
  MPI_Gather(local_part.data(), part_size, MPI_INT, v.data(), part_size, MPI_INT, 0, MPI_COMM_WORLD);
  if (proc_rank == 0 && padding_count > 0) {
    v.resize(enlarged_size - padding_count);
  }
}

void RadixSort(std::vector<int>& v) { MPI_RadixSort(v); }

bool RadixSortTask::ValidationImpl() {
  return !task_data->inputs.empty() && !task_data->outputs.empty() &&
         task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool RadixSortTask::PreProcessingImpl() {
  unsigned int count = task_data->inputs_count[0];
  int* in_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
  data_.assign(in_ptr, in_ptr + count);
  return true;
}

bool RadixSortTask::RunImpl() {
  RadixSort(data_);
  return true;
}

bool RadixSortTask::PostProcessingImpl() {
  int* out_ptr = reinterpret_cast<int*>(task_data->outputs[0]);
  std::ranges::copy(data_, out_ptr);
  return true;
}

}  // namespace muradov_k_radix_sort