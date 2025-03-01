#include "mpi/komshina_d_sort_radius_for_real_numbers_with_simple_merge/include/ops_mpi.hpp"

#include <mpi.h>
#include <algorithm>
#include <numeric>
#include <ranges>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

bool komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi::TestTaskMPI::PreProcessingImpl() {
  if (world_rank_ == 0) {
    auto* in_ptr = reinterpret_cast<double*>(task_data->inputs[0]);
    input_data_.insert(input_data_.end(), in_ptr, in_ptr + task_data->inputs_count[0]);
    std::ranges::sort(input_data_, std::greater<>{}); 
  }
  return true;
}

bool komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi::TestTaskMPI::ValidationImpl() {
  if (world_rank_ == 0) {
    return task_data->inputs_count == task_data->outputs_count;
  }
  return true;
}

bool komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi::TestTaskMPI::RunImpl() {
  ExecuteParallelSorting();
  return true;
}

bool komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi::TestTaskMPI::PostProcessingImpl() {
  if (world_rank_ == 0) {
    auto* out_ptr = reinterpret_cast<double*>(task_data->outputs[0]);
    std::ranges::copy(sortedData_, out_ptr);
  }
  return true;
}

void komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi::TestTaskMPI::ProcessAndSortSignedNumbers(
    std::vector<double>& data) {
  std::vector<double> positives;
  std::vector<double> negatives;
  positives.reserve(data.size());
  negatives.reserve(data.size());

  for (double num : data) {
    (num < 0 ? negatives : positives).push_back(std::fabs(num));
  }

  ApplyRadixSorting(positives);
  ApplyRadixSorting(negatives);

  for (double& num : negatives) {
    num = -num;
  }

  data.clear();
  data.reserve(positives.size() + negatives.size());
  data.insert(data.end(), negatives.rbegin(), negatives.rend());
  data.insert(data.end(), positives.begin(), positives.end());
}

void komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi::TestTaskMPI::ApplyRadixSorting(
    std::vector<double>& data) {
  const int total_bits = sizeof(double) * 8;
  std::vector<std::vector<double>> bins(2);
  bins[0].reserve(data.size());
  bins[1].reserve(data.size());

  std::vector<double> temp(data.size());
  temp.reserve(data.size());

  for (int bit = 0; bit < total_bits; ++bit) {
    for (double num : data) {
      uint64_t key = *reinterpret_cast<uint64_t*>(&num);
      bins[(key >> bit) & 1].push_back(num);
    }

    size_t index = 0;
    for (auto& bin : bins) {
      for (double num : bin) {
        temp[index++] = num;
      }
      bin.clear();
    }

    data.swap(temp);
  }
}

namespace komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi {

void komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi::TestTaskMPI::ExecuteParallelSorting() {
  int total_size = static_cast<int>((world_rank_ == 0) ? input_data_.size() : 0);
  MPI_Bcast(&total_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  int local_size = (total_size / size_) + ((world_rank_ < (total_size % size_)) ? 1 : 0);
  std::vector<double> local_data(local_size);

  std::vector<int> send_counts(size_), displacements(size_);
  if (world_rank_ == 0) {
    int base = total_size / size_, remainder = total_size % size_;
    for (int i = 0; i < size_; ++i) {
      send_counts[i] = base + (i < remainder);
      displacements[i] = (i > 0) ? (displacements[i - 1] + send_counts[i - 1]) : 0;
    }
  }

  MPI_Scatterv(input_data_.data(), send_counts.data(), displacements.data(), MPI_DOUBLE, local_data.data(), local_size,
               MPI_DOUBLE, 0, MPI_COMM_WORLD);

  ProcessAndSortSignedNumbers(local_data);

  std::vector<int> recv_counts(size_);
  std::vector<int> recv_displacements(size_);
  MPI_Gather(&local_size, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (world_rank_ == 0) {
    recv_displacements[0] = 0;
    for (int i = 1; i < size_; ++i) {
      recv_displacements[i] = recv_displacements[i - 1] + recv_counts[i - 1];
    }
    sortedData_.resize(total_size);
  }

  MPI_Gatherv(local_data.data(), local_size, MPI_DOUBLE, sortedData_.data(), recv_counts.data(),
              recv_displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

  if (world_rank_ == 0) {
    std::inplace_merge(sortedData_.begin(), sortedData_.begin() + recv_counts[0], sortedData_.end());
  }
  MPI_Barrier(MPI_COMM_WORLD);
}
}  // namespace komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi