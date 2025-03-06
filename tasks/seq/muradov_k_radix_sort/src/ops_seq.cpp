#include "seq/muradov_k_radix_sort/include/ops_seq.hpp"

#include <algorithm>
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
}  // namespace

void RadixSort(std::vector<int>& v) { SequentialRadixSort(v); }

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