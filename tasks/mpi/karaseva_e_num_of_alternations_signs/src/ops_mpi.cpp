#include "mpi/karaseva_e_num_of_alternations_signs/include/ops_mpi.hpp"

#include <mpi.h>

#include <cstddef>
#include <vector>

bool karaseva_e_num_of_alternations_signs_mpi::AlternatingSignsMPI::PreProcessingImpl() {
  if (world_.rank() == 0) {
    unsigned int input_size = task_data->inputs_count[0];
    auto *in_ptr = reinterpret_cast<int *>(task_data->inputs[0]);
    input_ = std::vector<int>(in_ptr, in_ptr + input_size);
  }
  total_ = 0;
  return true;
}

bool karaseva_e_num_of_alternations_signs_mpi::AlternatingSignsMPI::ValidationImpl() {
  if (world_.rank() == 0) {
    return task_data->inputs_count[0] >= 2 && task_data->outputs_count[0] == 1;
  }
  return true;
}

bool karaseva_e_num_of_alternations_signs_mpi::AlternatingSignsMPI::RunImpl() {
  int rank = world_.rank();
  int size = world_.size();

  unsigned int input_size = 0;
  if (rank == 0) {
    input_size = task_data->inputs_count[0];
  }
  MPI_Bcast(&input_size, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

  std::vector<int> recv_counts(size, static_cast<int>(input_size / size));
  for (int i = 0; i < static_cast<int>(input_size % size); ++i) {
    recv_counts[i]++;
  }

  std::vector<int> local_input(recv_counts[rank]);

  if (rank == 0) {
    int offset = recv_counts[0];
    for (int proc = 1; proc < size; ++proc) {
      MPI_Send(input_.data() + offset, recv_counts[proc], MPI_INT, proc, 0, MPI_COMM_WORLD);
      offset += recv_counts[proc];
    }
    local_input.assign(input_.begin(), input_.begin() + recv_counts[0]);
  } else {
    MPI_Recv(local_input.data(), recv_counts[rank], MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  std::vector<int> local_signs(local_input.size());
  for (size_t i = 0; i < local_input.size(); ++i) {
    local_signs[i] = (local_input[i] >= 0) ? 1 : -1;
  }

  int local_count = 0;
  for (size_t i = 1; i < local_signs.size(); ++i) {
    if (local_signs[i - 1] != local_signs[i]) {
      ++local_count;
    }
  }

  int left_neighbor = 0;
  if (rank > 0) {
    MPI_Recv(&left_neighbor, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    if (left_neighbor != local_signs.front()) {
      ++local_count;
    }
  }
  if (rank < size - 1) {
    MPI_Send(&local_signs.back(), 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD);
  }

  int global_count = 0;
  MPI_Reduce(&local_count, &global_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    total_ = global_count;
  }

  return true;
}

bool karaseva_e_num_of_alternations_signs_mpi::AlternatingSignsMPI::PostProcessingImpl() {
  if (world_.rank() == 0) {
    reinterpret_cast<int *>(task_data->outputs[0])[0] = total_;
  }
  return true;
}