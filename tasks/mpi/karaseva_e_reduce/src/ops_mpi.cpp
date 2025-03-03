#define OMPI_SKIP_MPICXX

#include "mpi/karaseva_e_reduce/include/ops_mpi.hpp"

#include <mpi.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/collectives/broadcast.hpp>
#include <boost/mpi/communicator.hpp>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace mpi = boost::mpi;

namespace {
template <typename T>
void ApplyOperation(void *inbuf, void *inoutbuf, int count, MPI_Op op) {
  auto *in = reinterpret_cast<T *>(inbuf);
  auto *inout = reinterpret_cast<T *>(inoutbuf);
  for (int i = 0; i < count; i++) {
    if (op == MPI_SUM) {
      inout[i] += in[i];
    } else if (op == MPI_MAX) {
      inout[i] = (inout[i] > in[i]) ? inout[i] : in[i];
    } else if (op == MPI_MIN) {
      inout[i] = (inout[i] < in[i]) ? inout[i] : in[i];
    } else {
      throw "Unsupported operation\n";
    }
  }
}
}  // namespace

namespace {
template <typename T>
int Reduce(void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm) {
  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  int typesize{};
  MPI_Type_size(datatype, &typesize);
  memcpy(recvbuf, sendbuf, count * typesize);

  int step = 1;
  while (step < size) {
    if ((rank % (2 * step)) == 0) {
      int src = rank + step;
      if (src < size) {
        std::vector<T> temp_buf(count);
        MPI_Recv(temp_buf.data(), count, datatype, src, 0, comm, MPI_STATUS_IGNORE);
        ApplyOperation<T>(temp_buf.data(), recvbuf, count, op);
      }
    } else {
      int dest = rank - step;
      MPI_Send(recvbuf, count, datatype, dest, 0, comm);
      break;
    }
    step *= 2;
  }

  return MPI_SUCCESS;
}
}  // namespace

template <typename T>
bool karaseva_e_reduce_mpi::TestTaskMPI<T>::PreProcessingImpl() {
  mpi::communicator world;
  int rank = world.rank();
  int size = world.size();

  std::cout << "Rank " << rank << " - PreProcessingImpl started\n";

  if (rank == 0) {
    input_size_ = task_data->inputs_count[0];
    input_.resize(input_size_);
    auto *input_data = reinterpret_cast<T *>(task_data->inputs[0]);
    std::memcpy(input_.data(), input_data, input_size_ * sizeof(T));

    std::cout << "Rank " << rank << " - Input data: \n";
    for (int i = 0; i < input_size_; ++i) {
      std::cout << input_[i] << " ";
    }
    std::cout << "\n";

    for (int proc = 1; proc < size; proc++) {
      world.send(proc, 0, input_size_);
    }
  } else {
    world.recv(0, 0, input_size_);
  }

  local_size_ = input_size_ / size;
  remel_ = input_size_ % size;

  // Calculate the start and end indices for each process
  int start = rank * local_size_ + std::min(rank, remel_);
  int end = start + local_size_ + (rank < remel_ ? 1 : 0);

  if (rank == 0) {
    // Process 0 takes its portion
    local_input_.assign(input_.begin() + start, input_.begin() + end);

    // Send data to other processes
    for (int proc = 1; proc < size; proc++) {
      int proc_start = proc * local_size_ + std::min(proc, remel_);
      int proc_end = proc_start + local_size_ + (proc < remel_ ? 1 : 0);
      world.send(proc, 0, input_.data() + proc_start, proc_end - proc_start);
    }
  } else {
    // Receive data from process 0
    local_input_.resize(end - start);
    world.recv(0, 0, local_input_.data(), end - start);
  }

  std::cout << "Rank " << rank << " - Local input: \n";
  for (const auto &val : local_input_) {
    std::cout << val << " ";
  }
  std::cout << "\n";

  return true;
}

template <typename T>
bool karaseva_e_reduce_mpi::TestTaskMPI<T>::ValidationImpl() {
  mpi::communicator world;
  std::cout << "Rank " << world.rank() << " - ValidationImpl started\n";
  return task_data->inputs_count[0] > 0 && task_data->outputs_count[0] == 1;
}

template <typename T>
bool karaseva_e_reduce_mpi::TestTaskMPI<T>::RunImpl() {
  mpi::communicator world;
  int rank = world.rank();

  std::cout << "Rank " << rank << " - RunImpl started\n";

  T local_sum = std::accumulate(local_input_.begin(), local_input_.end(), static_cast<T>(0));
  std::cout << "Rank " << rank << " - Local sum: " << local_sum << "\n";

  T global_sum = local_sum;

  MPI_Datatype mpi_type;
  if constexpr (std::is_same_v<T, int>) {
    mpi_type = MPI_INT;
  } else if constexpr (std::is_same_v<T, float>) {
    mpi_type = MPI_FLOAT;
  } else if constexpr (std::is_same_v<T, double>) {
    mpi_type = MPI_DOUBLE;
  } else {
    throw std::runtime_error("Unsupported type for MPI operation");
  }

  Reduce<T>(&local_sum, &global_sum, 1, mpi_type, MPI_SUM, 0, MPI_COMM_WORLD);

  // Broadcast
  MPI_Bcast(&global_sum, 1, mpi_type, 0, MPI_COMM_WORLD);

  result_ = global_sum;

  std::cout << "Rank " << rank << " - Final result: " << result_ << "\n";

  return true;
}

template <typename T>
bool karaseva_e_reduce_mpi::TestTaskMPI<T>::PostProcessingImpl() {
  mpi::communicator world;
  int rank = world.rank();

  std::cout << "Rank " << rank << " - PostProcessingImpl started\n";

  // Broadcast the result to all processes
  MPI_Bcast(&result_, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  // Ensure the output buffer is allocated
  if (task_data->outputs[0] == nullptr) {
    task_data->outputs[0] = new uint8_t[sizeof(T)];
  }

  // Write the result to the output buffer
  auto *output_ptr = reinterpret_cast<T *>(task_data->outputs[0]);
  *output_ptr = result_;

  std::cout << "Rank " << rank << " - Output value: " << *output_ptr << "\n";

  return true;
}

// Explicit instantiations
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