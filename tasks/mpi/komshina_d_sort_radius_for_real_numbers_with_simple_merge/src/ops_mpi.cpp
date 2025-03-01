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
    auto* input_ptr = reinterpret_cast<double*>(task_data->inputs[0]);
    input_data_.assign(input_ptr, input_ptr + task_data->inputs_count[0]);

    std::sort(input_data_.rbegin(), input_data_.rend());
  }
  return true;
}

bool komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi::TestTaskMPI::ValidationImpl() {
  if (rank == 0) {
    if (task_data->inputs_count.size() != task_data->outputs_count.size()) {
      return false;
    }

    for (size_t i = 0; i < task_data->inputs_count.size(); ++i) {
      if (task_data->inputs_count[i] != task_data->outputs_count[i]) {
        return false;
      }
    }
    return true;
  }
  return true;
}

bool komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi::TestTaskMPI::RunImpl() {
  parallelSort();
  return true;
}

bool komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi::TestTaskMPI::PostProcessingImpl() {
  if (rank == 0) {
    auto* output_ptr = reinterpret_cast<double*>(task_data->outputs[0]);
    std::copy(sorted_data_.begin(), sorted_data_.end(), output_ptr);
  }

  return true;
}
void komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi::radixSortWithSignHandling(
    std::vector<double>& data) {
  const int num_bits = sizeof(double) * 8;
  const int radix = 2;

  std::vector<double> positives;
  std::vector<double> negatives;

  for (double num : data) {
    if (num < 0) {
      negatives.push_back(-num);
    } else {
      positives.push_back(num);
    }
  }

  radixSort(positives, num_bits, radix);
  radixSort(negatives, num_bits, radix);

  for (double& num : negatives) {
    num = -num;
  }

  data.clear();
  data.insert(data.end(), negatives.rbegin(), negatives.rend());
  data.insert(data.end(), positives.begin(), positives.end());
}

void komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi::radixSort(std::vector<double>& data, int num_bits,
                                                                              int radix) {
  std::vector<std::vector<double>> buckets(radix);
  std::vector<double> output(data.size());

  for (int exp = 0; exp < num_bits; ++exp) {
    for (auto& num : data) {
      uint64_t bits = *reinterpret_cast<uint64_t*>(&num);
      int digit = (bits >> exp) & 1;
      buckets[digit].push_back(num);
    }

    int index = 0;
    for (int i = 0; i < radix; ++i) {
      for (auto& num : buckets[i]) {
        output[index++] = num;
      }
      buckets[i].clear();
    }

    data = output;
  }
}

void komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi::TestTaskMPI::parallelSort() {
  int total_size;
  if (rank == 0) total_size = input_data_.size();
  MPI_Bcast(&total_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<int> send_counts(size);
  std::vector<int> displacements(size);

  int base_size = total_size / size;
  int remainder = total_size % size;

  for (int i = 0; i < size; ++i) {
    send_counts[i] = base_size + (i < remainder ? 1 : 0);
    displacements[i] = (i > 0) ? (displacements[i - 1] + send_counts[i - 1]) : 0;
  }

  std::vector<double> local_data(send_counts[rank]);
  MPI_Scatterv(input_data_.data(), send_counts.data(), displacements.data(), MPI_DOUBLE, local_data.data(),
               send_counts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

  radixSortWithSignHandling(local_data);

  if (rank == 0) {
    std::vector<double> recv_data(base_size + 1);
    std::vector<double> merged_data;
    sorted_data_ = local_data;

    MPI_Status status;
    merged_data.reserve(total_size);
    for (int proc = 1; proc < size; proc++) {
      MPI_Recv(recv_data.data(), send_counts[proc], MPI_DOUBLE, proc, 0, MPI_COMM_WORLD, &status);
      std::merge(sorted_data_.begin(), sorted_data_.end(), recv_data.begin(), recv_data.begin() + send_counts[proc],
                 std::back_inserter(merged_data));
      sorted_data_ = merged_data;
      merged_data.clear();
    }
  } else {
    MPI_Send(local_data.data(), local_data.size(), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
  }

  MPI_Barrier(MPI_COMM_WORLD);
}