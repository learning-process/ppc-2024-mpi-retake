#include "seq/komshina_d_sort_radius_for_real_numbers_with_simple_merge/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

bool komshina_d_sort_radius_for_real_numbers_with_simple_merge_seq::TestTaskSequential::PreProcessingImpl() {
  auto* in_ptr = reinterpret_cast<double*>(task_data->inputs[0]);
  input_.insert(input_.end(), in_ptr, in_ptr + task_data->inputs_count[0]);
  return true;
}

bool komshina_d_sort_radius_for_real_numbers_with_simple_merge_seq::TestTaskSequential::ValidationImpl() {
  return task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool komshina_d_sort_radius_for_real_numbers_with_simple_merge_seq::TestTaskSequential::RunImpl() {
  SortWithSignHandling(input_);
  output_ = input_;
  return true;
}

bool komshina_d_sort_radius_for_real_numbers_with_simple_merge_seq::TestTaskSequential::PostProcessingImpl() {
  auto* out_ptr = reinterpret_cast<double*>(task_data->outputs[0]);
  std::ranges::copy(output_.begin(), output_.end(), out_ptr);
  return true;
}

void komshina_d_sort_radius_for_real_numbers_with_simple_merge_seq::TestTaskSequential::SortWithSignHandling(
    std::vector<double>& values) {
  std::vector<double> positives;
  std::vector<double> negatives;
  positives.reserve(values.size());
  negatives.reserve(values.size());

  for (double num : values) {
    (num < 0 ? negatives : positives).push_back(std::fabs(num));
  }

  BucketRadixSort(positives);
  BucketRadixSort(negatives);

  for (double& num : negatives) {
    num = -num;
  }

  values.clear();
  values.reserve(positives.size() + negatives.size());
  values.insert(values.end(), negatives.rbegin(), negatives.rend());
  values.insert(values.end(), positives.begin(), positives.end());
}

void komshina_d_sort_radius_for_real_numbers_with_simple_merge_seq::BucketRadixSort(std::vector<double>& values) {
  const int total_bits = sizeof(double) * 8;
  std::vector<std::vector<double>> bins(2);
  bins[0].reserve(values.size());
  bins[1].reserve(values.size());

  std::vector<double> temp(values.size());
  temp.reserve(values.size());

  for (int bit = 0; bit < total_bits; ++bit) {
    for (double num : values) {
      uint64_t key = 0;
      std::memcpy(&key, &num, sizeof(num));
      bins[(key >> bit) & 1].push_back(num);
    }

    size_t index = 0;
    for (auto& bin : bins) {
      for (double num : bin) {
        temp[index++] = num;
      }
      bin.clear();
    }

    values.swap(temp);
  }
}