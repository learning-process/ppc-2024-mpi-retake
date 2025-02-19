#include "mpi/karaseva_e_reduce/include/ops_mpi.hpp"

#include <mpi.h>

#include <cmath>
#include <cstring>
#include <numeric>
#include <vector>

namespace {

// Utility function to get MPI datatype based on template type
template <typename T>
MPI_Datatype GetMPIType();

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
  auto* in_ptr = reinterpret_cast<T*>(task_data->inputs[0]);
  input_ = std::vector<T>(in_ptr, in_ptr + input_size);

  // Output should be a single element
  output_.resize(1, static_cast<T>(0));

  return true;
}

template <typename T>
bool karaseva_e_reduce_mpi::TestTaskMPI<T>::ValidationImpl() {
  return task_data->inputs_count[0] > 0 && task_data->outputs_count[0] == 1;
}

template <typename T>
bool karaseva_e_reduce_mpi::TestTaskMPI<T>::RunImpl() {
  T local_sum = std::accumulate(input_.begin(), input_.end(), static_cast<T>(0));
  T global_sum = 0;

  int rank = 0, size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Binary tree reduction
  for (int step = 1; step < size; step *= 2) {
    int partner_rank = rank ^ step;
    if (partner_rank >= size) continue;

    if (rank < partner_rank) {
      T recv_data = 0;
      MPI_Recv(&recv_data, 1, GetMPIType<T>(), partner_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      local_sum += recv_data;
    } else {
      MPI_Send(&local_sum, 1, GetMPIType<T>(), partner_rank, 0, MPI_COMM_WORLD);
      break;
    }
  }

  if (rank == 0) {
    global_sum = local_sum;
    output_[0] = global_sum;
  }

  return true;
}

template <typename T>
bool karaseva_e_reduce_mpi::TestTaskMPI<T>::PostProcessingImpl() {
  if (task_data->outputs_count[0] > 0) {
    std::memcpy(task_data->outputs[0], output_.data(), sizeof(T));
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