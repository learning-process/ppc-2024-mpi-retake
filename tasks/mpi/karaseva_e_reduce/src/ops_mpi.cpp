#include "mpi/karaseva_e_reduce/include/ops_mpi.hpp"

#include <mpi.h>

#include <cmath>
#include <cstring>
#include <numeric>
#include <vector>

namespace {

// Utility function to get MPI datatype based on template type
template <typename T>
static MPI_Datatype GetMPIType();

template <>
MPI_Datatype GetMPIType<int>() {
  return MPI_INT;
}

template <>
MPI_Datatype GetMPIType<float>() {
  return MPI_FLOAT;
}

template <>
MPI_Datatype GetMPIType<double>() {
  return MPI_DOUBLE;
}

}  // namespace

template <typename T>
bool karaseva_e_reduce_mpi::TestTaskMPI<T>::PreProcessingImpl() {
  unsigned int input_size = task_data->inputs_count[0];
  auto* in_ptr = static_cast<T*>(static_cast<void*>(task_data->inputs[0]));
  input_ = std::vector<T>(in_ptr, in_ptr + input_size);

  unsigned int output_size = task_data->outputs_count[0];
  output_ = std::vector<T>(output_size, static_cast<T>(0));

  rc_size_ = static_cast<int>(std::sqrt(input_size));
  return true;
}

template <typename T>
bool karaseva_e_reduce_mpi::TestTaskMPI<T>::ValidationImpl() {
  return task_data->inputs_count[0] > 1 && task_data->outputs_count[0] == 1;
}

template <typename T>
bool karaseva_e_reduce_mpi::TestTaskMPI<T>::RunImpl() {
  T local_sum = std::accumulate(input_.begin(), input_.end(), static_cast<T>(0));
  T recv_data = 0;

  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  for (int step = 1; step < size; step *= 2) {
    int partner_rank = rank ^ step;

    if (partner_rank < size) {
      if (rank < partner_rank) {
        MPI_Recv(&recv_data, 1, GetMPIType<T>(), partner_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        local_sum += recv_data;
      } else {
        MPI_Send(&local_sum, 1, GetMPIType<T>(), partner_rank, 0, MPI_COMM_WORLD);
        break;
      }
    }
  }

  if (rank == 0) {
    output_[0] = local_sum;
  }

  MPI_Barrier(MPI_COMM_WORLD);
  return true;
}

template <typename T>
bool karaseva_e_reduce_mpi::TestTaskMPI<T>::PostProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (task_data->outputs_count[0] > 0 && rank == 0) {
    std::memcpy(static_cast<T*>(static_cast<void*>(task_data->outputs[0])), output_.data(), sizeof(T));
  }
  return true;
}

// Explicit template instantiations
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