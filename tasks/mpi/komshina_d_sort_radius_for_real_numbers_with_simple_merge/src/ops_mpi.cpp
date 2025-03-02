#include "mpi/komshina_d_sort_radius_for_real_numbers_with_simple_merge/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <cmath>
#include <cstddef>
#include <vector>

namespace mpi = boost::mpi;

bool komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi::TestTaskMPI::PreProcessingImpl() {
  if (world_.rank() == 0) {
    numbers_.resize(total_size_);
    std::memcpy(numbers_.data(), task_data->inputs[1], total_size_ * sizeof(double));
  }
  return true;
}

bool komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi::TestTaskMPI::ValidationImpl() {
  bool is_valid = world_.rank() == 0;
  if (is_valid) {
    total_size_ = *reinterpret_cast<int*>(task_data->inputs[0]);
    is_valid = task_data->inputs_count[0] == 1 && task_data->inputs_count[1] == static_cast<size_t>(total_size_) &&
               task_data->outputs_count[0] == static_cast<size_t>(total_size_);
  }
  mpi::broadcast(world_, is_valid, 0);
  mpi::broadcast(world_, total_size_, 0);
  return is_valid;
}

bool komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi::TestTaskMPI::RunImpl() {
  int rank = world_.rank();
  int size = world_.size();
  int chunk_size = total_size_ / size;
  int remainder = total_size_ % size;

  std::vector<int> sizes(size, chunk_size);
  for (int i = 0; i < remainder; ++i) sizes[i]++;

  std::vector<int> offsets(size, 0);
  for (int i = 1; i < size; ++i) offsets[i] = offsets[i - 1] + sizes[i - 1];

  std::vector<double> local_data(sizes[rank]);
  mpi::scatterv(world_, numbers_.data(), sizes, offsets, local_data.data(), sizes[rank], 0);
  SortDoubles(local_data);

  int step = 1;
  while (step < size) {
    if (rank % (2 * step) == 0) {
      int partner = rank + step;
      if (partner < size) {
        int partner_size;
        world_.recv(partner, 0, partner_size);
        std::vector<double> partner_data(partner_size);
        world_.recv(partner, 1, partner_data.data(), partner_size);
        std::vector<double> merged;
        merged.reserve(local_data.size() + partner_data.size());
        std::merge(local_data.begin(), local_data.end(), partner_data.begin(), partner_data.end(),
                   std::back_inserter(merged));
        local_data.swap(merged);
      }
    } else if (rank % (2 * step) == step) {
      world_.send(rank - step, 0, static_cast<int>(local_data.size()));
      world_.send(rank - step, 1, local_data.data(), static_cast<int>(local_data.size()));
      local_data.clear();
    }
    step *= 2;
  }

  if (rank == 0) numbers_.swap(local_data);
  return true;
}

bool komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi::TestTaskMPI::PostProcessingImpl() {
  if (world_.rank() == 0) {
    std::memcpy(task_data->outputs[0], numbers_.data(), total_size_ * sizeof(double));
  }
  return true;
}

namespace komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi {

void TestTaskMPI::SortDoubles(std::vector<double>& arr) {
  std::vector<uint64_t> keys(arr.size());
  const uint64_t sign_mask = (1ULL << 63);

  for (size_t i = 0; i < arr.size(); ++i) {
    uint64_t temp;
    std::memcpy(&temp, &arr[i], sizeof(double));
    temp = (temp & sign_mask) ? ~temp : (temp | sign_mask);
    keys[i] = temp;
  }

  SortUint64(keys);

  for (size_t i = 0; i < arr.size(); ++i) {
    uint64_t temp = keys[i];
    temp = (temp & sign_mask) ? (temp & ~sign_mask) : ~temp;
    std::memcpy(&arr[i], &temp, sizeof(double));
  }
}

void TestTaskMPI::SortUint64(std::vector<uint64_t>& keys) {
  constexpr int BIT_COUNT = 64;
  constexpr int BUCKET_COUNT = 256;
  std::vector<uint64_t> temp_buffer(keys.size());

  for (int shift = 0; shift < BIT_COUNT; shift += 8) {
    std::array<size_t, BUCKET_COUNT + 1> histogram{};

    for (uint64_t num : keys) {
      ++histogram[((num >> shift) & 0xFF) + 1];
    }

    for (int i = 0; i < BUCKET_COUNT; ++i) {
      histogram[i + 1] += histogram[i];
    }

    for (uint64_t num : keys) {
      temp_buffer[histogram[(num >> shift) & 0xFF]++] = num;
    }

    keys.swap(temp_buffer);
  }
}
}  // namespace komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi