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

std::vector<int> MergeTwoAscending(const std::vector<int>& a, const std::vector<int>& b) {
  std::vector<int> res(a.size() + b.size());
  std::size_t i = 0, j = 0, k = 0;
  while (i < a.size() && j < b.size()) {
    if (a[i] <= b[j])
      res[k++] = a[i++];
    else
      res[k++] = b[j++];
  }
  while (i < a.size()) res[k++] = a[i++];
  while (j < b.size()) res[k++] = b[j++];
  return res;
}

}  // anonymous namespace

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
  // Padding: on rank 0, compute pad value based on input range.
  int padding_count = 0;
  int pad_value = 0;
  bool pad_at_beginning = false;
  if (proc_rank == 0) {
    int min_val = v[0], max_val = v[0];
    for (int x : v) {
      if (x < min_val) min_val = x;
      if (x > max_val) max_val = x;
    }
    if (max_val < 0) {
      pad_value = min_val - 1;
      pad_at_beginning = true;
    } else {
      pad_value = max_val + 1;
      pad_at_beginning = false;
    }
    while (v.size() % proc_count != 0) {
      v.push_back(pad_value);
      ++padding_count;
    }
  }
  int enlarged_size = 0;
  if (proc_rank == 0) {
    enlarged_size = static_cast<int>(v.size());
  }
  MPI_Bcast(&enlarged_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
  int part_size = enlarged_size / proc_count;
  std::vector<int> local_array(part_size);
  MPI_Scatter(v.data(), part_size, MPI_INT, local_array.data(), part_size, MPI_INT, 0, MPI_COMM_WORLD);
  SequentialRadixSort(local_array);
  // Tree-based merge reduction.
  int current_size = part_size;
  for (int d = 1; d < proc_count; d *= 2) {
    if (proc_rank % (2 * d) == 0) {
      if (proc_rank + d < proc_count) {
        std::vector<int> recv_array(current_size);
        MPI_Status status;
        MPI_Recv(recv_array.data(), current_size, MPI_INT, proc_rank + d, 0, MPI_COMM_WORLD, &status);
        std::vector<int> merged = MergeTwoAscending(local_array, recv_array);
        local_array = merged;
        current_size *= 2;
      }
    } else {
      int target = proc_rank - (proc_rank % (2 * d));
      MPI_Send(local_array.data(), current_size, MPI_INT, target, 0, MPI_COMM_WORLD);
      break;
    }
  }
  if (proc_rank == 0) {
    // Remove padded elements.
    if (pad_at_beginning)
      v = std::vector<int>(local_array.begin() + padding_count, local_array.end());
    else
      v = std::vector<int>(local_array.begin(), local_array.end() - padding_count);
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