#include "mpi/komshina_d_sort_radius_for_real_numbers_with_simple_merge/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <vector>

bool komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi::TestTaskMPI::PreProcessingImpl() {
  if (world_.rank() == 0) {
    numbers_.resize(total_size_);
    std::memcpy(numbers_.data(), task_data->inputs[1], total_size_ * sizeof(double));
  }
  return true;
}

bool komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi::TestTaskMPI::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  bool is_valid = (rank == 0);

  if (is_valid) {
    total_size_ = *reinterpret_cast<int*>(task_data->inputs[0]);
    is_valid = task_data->inputs_count[0] == 1 && task_data->inputs_count[1] == static_cast<size_t>(total_size_) &&
               task_data->outputs_count[0] == static_cast<size_t>(total_size_);
  }

  MPI_Bcast(&is_valid, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
  MPI_Bcast(&total_size_, 1, MPI_INT, 0, MPI_COMM_WORLD);
  return is_valid;
}

bool komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi::TestTaskMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int chunk_size = total_size_ / size;
  int remainder = total_size_ % size;

  std::vector<int> sizes(size, chunk_size);
  for (int i = 0; i < remainder; ++i) {
    sizes[i]++;
  }

  std::vector<int> offsets(size, 0);
  for (int i = 1; i < size; ++i) {
    offsets[i] = offsets[i - 1] + sizes[i - 1];
  }

  std::vector<double> local_data(sizes[rank]);
  MPI_Scatterv(numbers_.data(), sizes.data(), offsets.data(), MPI_DOUBLE, local_data.data(), sizes[rank], MPI_DOUBLE, 0,
               MPI_COMM_WORLD);

  SortDoubles(local_data);

  int step = 1;
  while (step < size) {
    if (rank % (2 * step) == 0) {
      int partner = rank + step;
      if (partner < size) {
        int partner_size = 0;
        MPI_Recv(&partner_size, 1, MPI_INT, partner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        std::vector<double> partner_data(partner_size);
        MPI_Recv(partner_data.data(), partner_size, MPI_DOUBLE, partner, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        std::vector<double> merged;
        merged.reserve(local_data.size() + partner_data.size());
        std::ranges::merge(local_data, partner_data, std::back_inserter(merged));
        local_data.swap(merged);
      }
    } else if (rank % (2 * step) == step) {
      int local_size = static_cast<int>(local_data.size());
      MPI_Send(&local_size, 1, MPI_INT, rank - step, 0, MPI_COMM_WORLD);
      MPI_Send(local_data.data(), static_cast<int>(local_data.size()), MPI_DOUBLE, rank - step, 1, MPI_COMM_WORLD);
      local_data.clear();
    }
    step *= 2;
  }

  if (rank == 0) {
    numbers_.swap(local_data);
  }
  return true;
}

bool komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi::TestTaskMPI::PostProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    std::memcpy(task_data->outputs[0], numbers_.data(), total_size_ * sizeof(double));
  }
  return true;
}

namespace komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi {

void TestTaskMPI::SortDoubles(std::vector<double>& arr) {
  std::vector<uint64_t> keys(arr.size());
  const uint64_t sign_mask = (1ULL << 63);

  for (size_t i = 0; i < arr.size(); ++i) {
    uint64_t temp = 0;
    std::memcpy(&temp, &arr[i], sizeof(double));
    temp = ((temp & sign_mask) != 0) ? ~temp : (temp | sign_mask);
    keys[i] = temp;
  }

  SortUint64(keys);

  for (size_t i = 0; i < arr.size(); ++i) {
    uint64_t temp = keys[i];
    temp = ((temp & sign_mask) != 0) ? (temp & ~sign_mask) : ~temp;
    std::memcpy(&arr[i], &temp, sizeof(double));
  }
}

void TestTaskMPI::SortUint64(std::vector<uint64_t>& keys) {
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
}  // namespace komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi