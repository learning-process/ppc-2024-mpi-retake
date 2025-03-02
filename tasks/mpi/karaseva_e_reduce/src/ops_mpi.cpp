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

  input_.resize(input_size);
  std::memcpy(input_.data(), task_data->inputs[0], input_size * sizeof(T));

  unsigned int output_size = task_data->outputs_count[0];
  output_.resize(output_size, static_cast<T>(0));

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

  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  T global_sum = 0;
  MPI_Reduce(&local_sum, &global_sum, 1, GetMPIType<T>(), MPI_SUM, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    output_[0] = global_sum;
  }

  MPI_Barrier(MPI_COMM_WORLD);
  return true;
}

template <typename T>
bool karaseva_e_reduce_mpi::TestTaskMPI<T>::PostProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (task_data->outputs_count[0] > 0 && rank == 0) {
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