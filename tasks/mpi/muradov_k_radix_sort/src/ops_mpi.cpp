#define OMPI_SKIP_MPICXX
#include "mpi/muradov_k_radix_sort/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstdlib>
#include <ctime>
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
  if (arr.empty()) {
    return;
  }
  int max_val = *std::ranges::max_element(arr);
  for (int exp = 1; max_val / exp > 0; exp *= 10) {
    CountingSortForRadix(arr, exp);
  }
}

// seq
void SequentialRadixSort(std::vector<int>& v) {
  std::vector<int> negatives;
  std::vector<int> non_negatives;
  for (int x : v) {
    if (x < 0) {
      negatives.push_back(-x);
    } else {
      non_negatives.push_back(x);
    }
  }
  LSDRadixSort(non_negatives);
  LSDRadixSort(negatives);
  std::ranges::reverse(negatives);
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
  std::size_t i = 0;
  std::size_t j = 0;
  std::size_t k = 0;

  while (i < a.size() && j < b.size()) {
    if (a[i] <= b[j]) {
      res[k++] = a[i++];
    } else {
      res[k++] = b[j++];
    }
  }
  while (i < a.size()) {
    res[k++] = a[i++];
  }
  while (j < b.size()) {
    res[k++] = b[j++];
  }
  return res;
}

}  // anonymous namespace

void MpiRadixSort(std::vector<int>& v) {
  int proc_rank = 0;
  int proc_count = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &proc_count);
  if (proc_count <= 1 || static_cast<int>(v.size()) <= proc_count) {
    if (proc_rank == 0 && !v.empty()) {
      SequentialRadixSort(v);
    }
    return;
  }
  // Padding on rank 0.
  int padding_count = 0;
  int pad_value = 0;
  bool pad_at_beginning = false;
  if (proc_rank == 0) {
    int min_val = v[0];
    int max_val = v[0];
    for (int x : v) {
      min_val = std::min(x, min_val);
      max_val = std::max(x, max_val);
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
  // Tree-based merge reduction (binary tree reduction handling variable sizes)
  int my_size = part_size;
  int step = 1;
  while (step < proc_count) {
    if (proc_rank % (2 * step) == 0) {
      int src = proc_rank + step;
      if (src < proc_count) {
        int recv_size = 0;
        MPI_Recv(&recv_size, 1, MPI_INT, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        std::vector<int> recv_array(recv_size);
        MPI_Recv(recv_array.data(), recv_size, MPI_INT, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        std::vector<int> merged = MergeTwoAscending(local_array, recv_array);
        local_array = merged;
        my_size = static_cast<int>(local_array.size());
      }
    } else {
      int target = proc_rank - (proc_rank % (2 * step));
      MPI_Send(&my_size, 1, MPI_INT, target, 0, MPI_COMM_WORLD);
      MPI_Send(local_array.data(), my_size, MPI_INT, target, 0, MPI_COMM_WORLD);
      break;
    }
    step *= 2;
  }
  if (proc_rank == 0) {
    if (pad_at_beginning) {
      v = std::vector<int>(local_array.begin() + padding_count, local_array.end());
    } else {
      v = std::vector<int>(local_array.begin(), local_array.end() - padding_count);
    }
  }
}

void RadixSort(std::vector<int>& v) { MpiRadixSort(v); }

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
