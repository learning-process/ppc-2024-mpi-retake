#include "mpi/komshina_d_sort_radius_for_real_numbers_with_simple_merge/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
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
  std::vector<double> positives, negatives;
  for (double num : data) {
    (num < 0 ? negatives : positives).push_back(std::abs(num));
  }

  ApplyRadixSorting(positives);
  ApplyRadixSorting(negatives);

  for (double& num : negatives) num = -num;

  data.clear();
  data.insert(data.end(), negatives.rbegin(), negatives.rend());
  data.insert(data.end(), positives.begin(), positives.end());
}

void komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi::TestTaskMPI::ApplyRadixSorting(
    std::vector<double>& data) {
  constexpr int BITS = sizeof(double) * 8;
  constexpr int RADIX = 2;
  std::vector<std::vector<double>> buckets(RADIX);
  std::vector<double> temp(data.size());

  for (int exp = 0; exp < BITS; ++exp) {
    for (double num : data) {
      uint64_t bits = *reinterpret_cast<uint64_t*>(&num);
      buckets[(bits >> exp) & 1].push_back(num);
    }

    size_t idx = 0;
    for (auto& bucket : buckets) {
      for (double num : bucket) temp[idx++] = num;
      bucket.clear();
    }
    data.swap(temp);
  }
}

namespace komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi {

void komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi::TestTaskMPI::ExecuteParallelSorting() {
  int total_size = (world_rank_ == 0) ? input_data_.size() : 0;
  MPI_Bcast(&total_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<int> send_counts(size_), displacements(size_);
  int base = total_size / size_, remainder = total_size % size_;

  for (int i = 0; i < size_; ++i) {
    send_counts[i] = base + (i < remainder);
    displacements[i] = (i > 0) ? (displacements[i - 1] + send_counts[i - 1]) : 0;
  }

  std::vector<double> local_data(send_counts[world_rank_]);
  MPI_Scatterv(input_data_.data(), send_counts.data(), displacements.data(), MPI_DOUBLE, local_data.data(),
               send_counts[world_rank_], MPI_DOUBLE, 0, MPI_COMM_WORLD);

  ProcessAndSortSignedNumbers(local_data);

  if (world_rank_ == 0) {
    sortedData_ = std::move(local_data);
    std::vector<double> recv_data;
    MPI_Status status;

    for (int proc = 1; proc < size_; ++proc) {
      recv_data.resize(send_counts[proc]);
      MPI_Recv(recv_data.data(), send_counts[proc], MPI_DOUBLE, proc, 0, MPI_COMM_WORLD, &status);
      std::vector<double> merged;
      std::merge(sortedData_.begin(), sortedData_.end(), recv_data.begin(), recv_data.end(),
                 std::back_inserter(merged));
      sortedData_ = std::move(merged);
    }
  } else {
    MPI_Send(local_data.data(), local_data.size(), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
  }
  MPI_Barrier(MPI_COMM_WORLD);
}
}  // namespace komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi