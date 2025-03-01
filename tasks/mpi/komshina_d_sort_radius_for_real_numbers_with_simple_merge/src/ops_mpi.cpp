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
  if (rank == 0) {
    double* input_ptr = reinterpret_cast<double*>(task_data->inputs[0]);
    input_data_.assign(input_ptr, input_ptr + task_data->inputs_count[0]);
    std::sort(input_data_.rbegin(), input_data_.rend());
  }
  return true;
}

bool komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi::TestTaskMPI::ValidationImpl() {
  return (rank != 0 || task_data->inputs_count == task_data->outputs_count);
}

bool komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi::TestTaskMPI::RunImpl() {
  parallelSort();
  return true;
}

bool komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi::TestTaskMPI::PostProcessingImpl() {
  if (rank == 0) {
    std::copy(sorted_data_.begin(), sorted_data_.end(), reinterpret_cast<double*>(task_data->outputs[0]));
  }
  return true;
}

void komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi::TestTaskMPI::handleSignAndSort(
    std::vector<double>& data) {
  std::vector<double> positives, negatives;
  for (double num : data) {
    (num < 0 ? negatives : positives).push_back(std::abs(num));
  }
  radixSort(positives);
  radixSort(negatives);
  for (double& num : negatives) num = -num;
  data.assign(negatives.rbegin(), negatives.rend());
  data.insert(data.end(), positives.begin(), positives.end());
}

void komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi::TestTaskMPI::radixSort(
    std::vector<double>& data) {
  constexpr int num_bits = sizeof(double) * 8;
  constexpr int radix = 2;
  std::vector<std::vector<double>> buckets(radix);
  std::vector<double> output(data.size());

  for (int exp = 0; exp < num_bits; ++exp) {
    for (double num : data) {
      int digit = (*reinterpret_cast<uint64_t*>(&num) >> exp) & 1;
      buckets[digit].push_back(num);
    }
    data.clear();
    for (auto& bucket : buckets) {
      data.insert(data.end(), bucket.begin(), bucket.end());
      bucket.clear();
    }
  }
}

namespace komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi {

void komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi::TestTaskMPI::parallelSort() {
  int total_size = (rank == 0) ? input_data_.size() : 0;
  MPI_Bcast(&total_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<int> send_counts(size), displacements(size);
  int base_size = total_size / size, remainder = total_size % size;
  for (int i = 0; i < size; ++i) {
    send_counts[i] = base_size + (i < remainder);
    displacements[i] = (i == 0) ? 0 : (displacements[i - 1] + send_counts[i - 1]);
  }

  std::vector<double> local_data(send_counts[rank]);
  MPI_Scatterv(input_data_.data(), send_counts.data(), displacements.data(), MPI_DOUBLE, local_data.data(),
               send_counts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

  handleSignAndSort(local_data);

  if (rank == 0) {
    sorted_data_ = local_data;
    std::vector<double> recv_data(base_size + 1), merged_data;
    MPI_Status status;
    for (int proc = 1; proc < size; proc++) {
      MPI_Recv(recv_data.data(), send_counts[proc], MPI_DOUBLE, proc, 0, MPI_COMM_WORLD, &status);
      merged_data.resize(sorted_data_.size() + send_counts[proc]);
      std::merge(sorted_data_.begin(), sorted_data_.end(), recv_data.begin(), recv_data.begin() + send_counts[proc],
                 merged_data.begin());
      sorted_data_.swap(merged_data);
    }
  } else {
    MPI_Send(local_data.data(), local_data.size(), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
  }
  MPI_Barrier(MPI_COMM_WORLD);
}
}  // namespace komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi