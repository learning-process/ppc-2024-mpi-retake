#include "seq/komshina_d_sort_radius_for_real_numbers_with_simple_merge/include/ops_seq.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

using namespace komshina_d_sort_radius_for_real_numbers_with_simple_merge_seq;

bool TestTaskSequential::PreProcessingImpl() {
  values_.assign(size_, 0.0);
  auto* input = reinterpret_cast<double*>(task_data->inputs[1]);
  std::memcpy(values_.data(), input, size_ * sizeof(double));
  return true;
}

bool TestTaskSequential::ValidationImpl() {
  size_ = *reinterpret_cast<int*>(task_data->inputs[0]);
  return task_data->inputs_count[0] == 1 && task_data->inputs_count[1] == static_cast<size_t>(size_) &&
         task_data->outputs_count[0] == static_cast<size_t>(size_);
}

bool TestTaskSequential::RunImpl() {
  SortValues(values_);
  return true;
}

bool TestTaskSequential::PostProcessingImpl() {
  auto* output = reinterpret_cast<double*>(task_data->outputs[0]);
  std::memcpy(output, values_.data(), values_.size() * sizeof(double));
  return true;
}

void TestTaskSequential::SortValues(std::vector<double>& values) {
  std::vector<uint64_t> encoded(values.size());
  const uint64_t sign_mask = (1ULL << 63);

  for (size_t i = 0; i < values.size(); ++i) {
    uint64_t temp = 0;
    std::memcpy(&temp, &values[i], sizeof(double));
    temp = ((temp & sign_mask) != 0) ? ~temp : (temp | sign_mask);
    encoded[i] = temp;
  }

  RadixSort(encoded);

  for (size_t i = 0; i < values.size(); ++i) {
    uint64_t temp = encoded[i];
    temp = ((temp & sign_mask) != 0) ? (temp & ~sign_mask) : ~temp;
    std::memcpy(&values[i], &temp, sizeof(double));
  }
}

void TestTaskSequential::RadixSort(std::vector<uint64_t>& keys) {
  constexpr int kBitCount = 64;
  constexpr int kBucketCount = 256;

  std::vector<uint64_t> temp_buffer(keys.size());

  for (int shift = 0; shift < kBitCount; shift += 8) {
    std::array<size_t, kBucketCount + 1> histogram{};

    for (uint64_t num : keys) {
      ++histogram[((num >> shift) & 0xFF) + 1];
    }

    for (int i = 0; i < kBucketCount; ++i) {
      histogram[i + 1] += histogram[i];
    }

    for (uint64_t num : keys) {
      temp_buffer[histogram[(num >> shift) & 0xFF]++] = num;
    }

    keys.swap(temp_buffer);
  }
}