#include "mpi/karaseva_e_reduce/include/ops_mpi.hpp"

#include <mpi.h>

#include <cmath>
#include <cstddef>
#include <numeric>
#include <type_traits>
#include <vector>

template <typename T>
bool karaseva_e_reduce_mpi::TestTaskMPI<T>::PreProcessingImpl() {
  // Read input data as T (int, float or double)
  unsigned int input_size = task_data->inputs_count[0];
  auto* in_ptr = reinterpret_cast<T*>(task_data->inputs[0]);
  input_ = std::vector<T>(in_ptr, in_ptr + input_size);

  // Initialize output vector, but only process with rank 0 will store the final result
  unsigned int output_size = task_data->outputs_count[0];
  output_ = std::vector<T>(output_size, 0.0);

  rc_size_ = static_cast<int>(std::sqrt(input_size));
  return true;
}

template <typename T>
bool karaseva_e_reduce_mpi::TestTaskMPI<T>::ValidationImpl() {
  // Validation ensures there is enough data to reduce, and the output array is of size 1
  return task_data->inputs_count[0] > 1 && task_data->outputs_count[0] == 1;
}

template <typename T>
bool karaseva_e_reduce_mpi::TestTaskMPI<T>::RunImpl() {
  T local_sum = std::accumulate(input_.begin(), input_.end(), T(0));

  // To store the global sum
  T global_sum = 0;

  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int partner_rank = 0;
  // Binary tree reduce (log(size) communication steps)
  for (int step = 1; step < size; step *= 2) {
    partner_rank = rank ^ step;

    if (rank < partner_rank) {
      // The smaller process sends the local_sum to the larger process
      if constexpr (std::is_same_v<T, float>) {
        MPI_Send(&local_sum, 1, MPI_FLOAT, partner_rank, 0, MPI_COMM_WORLD);
      } else if constexpr (std::is_same_v<T, double>) {
        MPI_Send(&local_sum, 1, MPI_DOUBLE, partner_rank, 0, MPI_COMM_WORLD);
      } else if constexpr (std::is_same_v<T, int>) {
        MPI_Send(&local_sum, 1, MPI_INT, partner_rank, 0, MPI_COMM_WORLD);
      }
      break;
    }

    if (rank > partner_rank) {
      // Larger process receives the local_sum from the smaller process
      T recv_data = 0;
      if constexpr (std::is_same_v<T, float>) {
        MPI_Recv(&recv_data, 1, MPI_FLOAT, partner_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      } else if constexpr (std::is_same_v<T, double>) {
        MPI_Recv(&recv_data, 1, MPI_DOUBLE, partner_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      } else if constexpr (std::is_same_v<T, int>) {
        MPI_Recv(&recv_data, 1, MPI_INT, partner_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      }
      local_sum += recv_data;
    }
  }

  // Only process 0 will store the final result
  if (rank == 0) {
    global_sum = local_sum;
    output_ = {global_sum};  // Store the result in output
  }

  MPI_Barrier(MPI_COMM_WORLD);  // Synchronize processes
  return true;
}

template <typename T>
bool karaseva_e_reduce_mpi::TestTaskMPI<T>::PostProcessingImpl() {
  // Only the root process writes to the output array
  if (task_data->outputs_count[0] > 0) {
    reinterpret_cast<T*>(task_data->outputs[0])[0] = output_[0];
  }
  return true;
}

// Explicit definition of template functions
template bool karaseva_e_reduce_mpi::TestTaskMPI<int>::PreProcessingImpl();
template bool karaseva_e_reduce_mpi::TestTaskMPI<int>::ValidationImpl();
template bool karaseva_e_reduce_mpi::TestTaskMPI<int>::RunImpl();
template bool karaseva_e_reduce_mpi::TestTaskMPI<int>::PostProcessingImpl();

template bool karaseva_e_reduce_mpi::TestTaskMPI<float>::PreProcessingImpl();
template bool karaseva_e_reduce_mpi::TestTaskMPI<float>::ValidationImpl();
template bool karaseva_e_reduce_mpi::TestTaskMPI<float>::RunImpl();
template bool karaseva_e_reduce_mpi::TestTaskMPI<float>::PostProcessingImpl();

template bool karaseva_e_reduce_mpi::TestTaskMPI<double>::PreProcessingImpl();
template bool karaseva_e_reduce_mpi::TestTaskMPI<double>::ValidationImpl();
template bool karaseva_e_reduce_mpi::TestTaskMPI<double>::RunImpl();
template bool karaseva_e_reduce_mpi::TestTaskMPI<double>::PostProcessingImpl();