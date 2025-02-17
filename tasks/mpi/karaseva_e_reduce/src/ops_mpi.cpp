#include "mpi/karaseva_e_reduce/include/ops_mpi.hpp"

#include <mpi.h>

#include <cmath>
#include <cstddef>
#include <numeric>
#include <vector>


template <typename T>
bool karaseva_e_reduce_mpi::TestTaskMPI<T>::PreProcessingImpl() {
  // We read the input data as T (int, float or double)
  unsigned int input_size = task_data->inputs_count[0];
  auto* in_ptr = reinterpret_cast<T*>(task_data->inputs[0]);
  input_ = std::vector<T>(in_ptr, in_ptr + input_size);

  // The output vector uses type T
  unsigned int output_size = task_data->outputs_count[0];
  output_ = std::vector<T>(output_size, 0.0);

  rc_size_ = static_cast<int>(std::sqrt(input_size));
  return true;
}

template <typename T>
bool karaseva_e_reduce_mpi::TestTaskMPI<T>::ValidationImpl() {
  // For reduction, the size of the input data must be greater than 1, and the size of the output data must be 1.
  return task_data->inputs_count[0] > 1 && task_data->outputs_count[0] == 1;
}

template <typename T>
bool karaseva_e_reduce_mpi::TestTaskMPI<T>::RunImpl() {
  // Local summation
  T local_sum = std::accumulate(input_.begin(), input_.end(), T(0));

  T global_sum = 0;

  // Binary tree for reduction
  int rank;
  int size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int partner_rank;
  for (int step = 1; step < size; step *= 2) {
    partner_rank = rank ^ step;

    if (rank < partner_rank) {
      // The smaller process sends the data
      MPI_Send(&local_sum, 1, MPI_DOUBLE, partner_rank, 0, MPI_COMM_WORLD);
      break;
    } else if (rank > partner_rank) {
      // A larger process receives the data
      T recv_data = 0;
      MPI_Recv(&recv_data, 1, MPI_DOUBLE, partner_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      local_sum += recv_data;
    }
  }

  if (rank == 0) {
    global_sum = local_sum;
    output_ = {global_sum};
  }

  MPI_Barrier(MPI_COMM_WORLD);
  return true;
}

template <typename T>
bool karaseva_e_reduce_mpi::TestTaskMPI<T>::PostProcessingImpl() {
  // Copying the result to the output array
  // Transform the output vector to match the type T
  for (size_t i = 0; i < output_.size(); i++) {
    reinterpret_cast<T*>(task_data->outputs[0])[i] = output_[i];
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