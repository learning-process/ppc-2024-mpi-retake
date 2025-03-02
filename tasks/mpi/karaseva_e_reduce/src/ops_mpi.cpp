#include "mpi/karaseva_e_reduce/include/ops_mpi.hpp"

#include <mpi.h>

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
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Ensure we calculate the input_size correctly
  if (rank == 0) {
    input_size_ = task_data->inputs_count[0];
    input_.resize(input_size_);
    std::memcpy(input_.data(), task_data->inputs[0], input_size_ * sizeof(T));
  }

  // Broadcast input size to all processes
  MPI_Bcast(&input_size_, 1, MPI_INT, 0, MPI_COMM_WORLD);

  // Calculate local sizes for each process
  local_size_ = input_size_ / size;
  remel_ = input_size_ % size;

  // Ensure data is sent correctly to each process
  if (rank == 0) {
    local_input_.assign(input_.begin(), input_.begin() + local_size_ + remel_);
    for (int proc = 1; proc < size; proc++) {
      // Send the correct portion of data to each process
      MPI_Send(input_.data() + remel_ + (proc * local_size_), local_size_, GetMPIType<T>(), proc, 0, MPI_COMM_WORLD);
    }
  } else {
    local_input_.resize(local_size_);
    MPI_Recv(local_input_.data(), local_size_, GetMPIType<T>(), 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  return true;
}

template <typename T>
bool karaseva_e_reduce_mpi::TestTaskMPI<T>::ValidationImpl() {
  return task_data->inputs_count[0] > 0 && task_data->outputs_count[0] == 1;
}

template <typename T>
bool karaseva_e_reduce_mpi::TestTaskMPI<T>::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  T local_sum = std::accumulate(local_input_.begin(), local_input_.end(), static_cast<T>(0));

  // Reduce
  if (rank == 0) {
    result_ = local_sum;
    for (int proc = 1; proc < size; proc++) {
      T recv_sum;
      MPI_Recv(&recv_sum, 1, GetMPIType<T>(), proc, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      result_ += recv_sum;
    }
  } else {
    MPI_Send(&local_sum, 1, GetMPIType<T>(), 0, 0, MPI_COMM_WORLD);
  }

  return true;
}

template <typename T>
bool karaseva_e_reduce_mpi::TestTaskMPI<T>::PostProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    T* output_ptr = reinterpret_cast<T*>(task_data->outputs[0]);
    *output_ptr = result_;
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