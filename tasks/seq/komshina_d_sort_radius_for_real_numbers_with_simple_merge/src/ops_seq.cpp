#include "seq/komshina_d_sort_radius_for_real_numbers_with_simple_merge/include/ops_seq.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

using namespace komshina_d_sort_radius_for_real_numbers_with_simple_merge_seq;

bool TestTaskSequential::PreProcessingImpl() {
  values.assign(size, 0.0);
  auto* input = reinterpret_cast<double*>(task_data->inputs[1]);
  std::memcpy(values.data(), input, size * sizeof(double));
  return true;
}

bool TestTaskSequential::ValidationImpl() {
  size = *reinterpret_cast<int*>(task_data->inputs[0]);
  return task_data->inputs_count[0] == 1 && task_data->inputs_count[1] == static_cast<size_t>(size) &&
         task_data->outputs_count[0] == static_cast<size_t>(size);
}

bool TestTaskSequential::RunImpl() {
  SortValues(values);
  return true;
}

bool TestTaskSequential::PostProcessingImpl() {
  auto* output = reinterpret_cast<double*>(task_data->outputs[0]);
  std::memcpy(output, values.data(), values.size() * sizeof(double));
  return true;
}

void TestTaskSequential::SortValues(std::vector<double>& values) {
  std::vector<uint64_t> encoded(values.size());
  const uint64_t sign_mask = (1ULL << 63);

  for (size_t i = 0; i < values.size(); ++i) {
    uint64_t temp;
    std::memcpy(&temp, &values[i], sizeof(double));
    temp = (temp & sign_mask) ? ~temp : (temp | sign_mask);
    encoded[i] = temp;
  }

  RadixSort(encoded);

  for (size_t i = 0; i < values.size(); ++i) {
    uint64_t temp = encoded[i];
    temp = (temp & sign_mask) ? (temp & ~sign_mask) : ~temp;
    std::memcpy(&values[i], &temp, sizeof(double));
  }
}

void TestTaskSequential::RadixSort(std::vector<uint64_t>& data) {
  constexpr int BIT_COUNT = 64;
  constexpr int BUCKET_COUNT = 256;
  std::vector<uint64_t> temp_buffer(data.size());

  for (int shift = 0; shift < BIT_COUNT; shift += 8) {
    std::array<size_t, BUCKET_COUNT + 1> histogram{};

    for (uint64_t num : data) {
      ++histogram[((num >> shift) & 0xFF) + 1];
    }

    for (int i = 0; i < BUCKET_COUNT; ++i) {
      histogram[i + 1] += histogram[i];
    }

    for (uint64_t num : data) {
      temp_buffer[histogram[(num >> shift) & 0xFF]++] = num;
    }

    data.swap(temp_buffer);
  }
}