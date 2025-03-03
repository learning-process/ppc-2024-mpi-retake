#define OMPI_SKIP_MPICXX

#include "mpi/karaseva_e_reduce/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <iostream>
#include <numeric>
#include <vector>

namespace mpi = boost::mpi;

namespace karaseva_e_reduce_mpi {

template <typename T>
bool TestTaskMPI<T>::PreProcessingImpl() {
  mpi::communicator world;
  const int rank = world.rank();
  const int size = world.size();

  if (rank == 0) {
    input_size_ = task_data->inputs_count[0];
    if (input_size_ <= 0) {
      throw std::runtime_error("Input size must be positive");
    }

    input_.resize(input_size_);
    auto* input_data = reinterpret_cast<T*>(task_data->inputs[0]);
    std::copy(input_data, input_data + input_size_, input_.begin());

    const int base_chunk = input_size_ / size;
    const int remainder = input_size_ % size;

    std::vector<int> counts(size, base_chunk);
    for (int i = 0; i < remainder; ++i) counts[i]++;

    int offset = 0;
    for (int proc = 0; proc < size; ++proc) {
      if (proc == 0) {
        local_input_.assign(input_.begin(), input_.begin() + counts[0]);
      } else {
        world.send(proc, 0, &input_[offset], counts[proc]);
      }
      offset += counts[proc];
    }
  } else {
    int recv_size;
    world.recv(0, 0, recv_size);
    local_input_.resize(recv_size);
    world.recv(0, 0, local_input_.data(), recv_size);
  }

  return true;
}

template <typename T>
bool TestTaskMPI<T>::ValidationImpl() {
  return task_data->inputs_count[0] > 0 && task_data->outputs_count[0] >= 1;
}

template <typename T>
bool TestTaskMPI<T>::RunImpl() {
  mpi::communicator world;
  const int rank = world.rank();

  T local_sum = std::accumulate(local_input_.begin(), local_input_.end(), T{0});

  MPI_Datatype mpi_type = MPI_DATATYPE_NULL;
  if constexpr (std::is_same_v<T, int>) {
    mpi_type = MPI_INT;
  } else if constexpr (std::is_same_v<T, float>) {
    mpi_type = MPI_FLOAT;
  } else if constexpr (std::is_same_v<T, double>) {
    mpi_type = MPI_DOUBLE;
  }

  T global_sum = 0;
  MPI_Reduce(&local_sum, &global_sum, 1, mpi_type, MPI_SUM, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    result_ = global_sum;
  }

  return true;
}

template <typename T>
bool TestTaskMPI<T>::PostProcessingImpl() {
  mpi::communicator world;
  const int rank = world.rank();

  if (rank == 0) {
    if (task_data->outputs[0] == nullptr) {
      task_data->outputs[0] = new uint8_t[sizeof(T)];
    }
    *reinterpret_cast<T*>(task_data->outputs[0]) = result_;
  }

  return true;
}

template class TestTaskMPI<int>;
template class TestTaskMPI<float>;
template class TestTaskMPI<double>;

}  // namespace karaseva_e_reduce_mpi