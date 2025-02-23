#include "seq/kavtorev_d_radix_double_sort/include/ops_seq.hpp"

#include <cmath>
#include <queue>

using namespace kavtorev_d_radix_double_sort;

bool RadixSortSequential::PreProcessingImpl() {
  data_.resize(n);
  auto* arr = reinterpret_cast<double*>(task_data->inputs[1]);
  std::copy(arr, arr + n, data_.begin());

  return true;
}

bool RadixSortSequential::ValidationImpl() {
  bool is_valid = true;
  n = *(reinterpret_cast<int*>(task_data->inputs[0]));
  if (task_data->inputs_count[0] != 1 || task_data->inputs_count[1] != static_cast<size_t>(n) ||
      task_data->outputs_count[0] != static_cast<size_t>(n)) {
    is_valid = false;
  }

  return is_valid;
}

bool RadixSortSequential::RunImpl() {
  radix_sort_doubles(data_);
  return true;
}

bool RadixSortSequential::PostProcessingImpl() {
  auto* out = reinterpret_cast<double*>(task_data->outputs[0]);
  std::copy(data_.begin(), data_.end(), out);
  return true;
}

void RadixSortSequential::radix_sort_doubles(std::vector<double>& data_) {
  size_t n_ = data_.size();
  std::vector<uint64_t> keys(n_);
  for (size_t i = 0; i < n_; ++i) {
    uint64_t u;
    std::memcpy(&u, &data_[i], sizeof(double));
    if ((u & 0x8000000000000000ULL) != 0) {
      u = ~u;
    } else {
      u |= 0x8000000000000000ULL;
    }
    keys[i] = u;
  }

  radix_sort_uint64(keys);

  for (size_t i = 0; i < n_; ++i) {
    uint64_t u = keys[i];
    if ((u & 0x8000000000000000ULL) != 0) {
      u &= ~0x8000000000000000ULL;
    } else {
      u = ~u;
    }
    std::memcpy(&data_[i], &u, sizeof(double));
  }
}

void RadixSortSequential::radix_sort_uint64(std::vector<uint64_t>& keys) {
  const int BITS = 64;
  const int RADIX = 256;
  std::vector<uint64_t> temp(keys.size());

  for (int shift = 0; shift < BITS; shift += 8) {
    size_t count[RADIX + 1] = {0};
    for (size_t i = 0; i < keys.size(); ++i) {
      uint8_t byte = (keys[i] >> shift) & 0xFF;
      ++count[byte + 1];
    }
    for (int i = 0; i < RADIX; ++i) {
      count[i + 1] += count[i];
    }
    for (size_t i = 0; i < keys.size(); ++i) {
      uint8_t byte = (keys[i] >> shift) & 0xFF;
      temp[count[byte]++] = keys[i];
    }
    keys.swap(temp);
  }
}