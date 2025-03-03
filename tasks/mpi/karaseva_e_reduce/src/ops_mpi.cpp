#define OMPI_SKIP_MPICXX

#include "mpi/karaseva_e_reduce/include/ops_mpi.hpp"

#include <boost/mpi/communicator.hpp>
#include <boost/mpi.hpp>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <vector>

namespace mpi = boost::mpi;

template <typename T>
bool karaseva_e_reduce_mpi::TestTaskMPI<T>::PreProcessingImpl() {
  mpi::communicator world;
  int rank = world.rank();
  int size = world.size();

  if (rank == 0) {
    input_size_ = task_data->inputs_count[0];
    input_.resize(input_size_);
    auto* input_data = reinterpret_cast<T*>(task_data->inputs[0]);
    std::memcpy(input_.data(), input_data, input_size_ * sizeof(T));

    for (int proc = 1; proc < size; proc++) {
      world.send(proc, 0, input_size_);
    }
  } else {
    world.recv(0, 0, input_size_);
  }

  local_size_ = input_size_ / size;
  remel_ = input_size_ % size;

  if (rank == 0) {
    local_input_.assign(input_.begin(), input_.begin() + local_size_ + remel_);
    for (int proc = 1; proc < size; proc++) {
      world.send(proc, 0, input_.data() + remel_ + (proc * local_size_), local_size_);
    }
  } else {
    local_input_.resize(local_size_);
    world.recv(0, 0, local_input_.data(), local_size_);
  }

  return true;
}

template <typename T>
bool karaseva_e_reduce_mpi::TestTaskMPI<T>::ValidationImpl() {
  return task_data->inputs_count[0] > 0 && task_data->outputs_count[0] == 1;
}

template <typename T>
bool karaseva_e_reduce_mpi::TestTaskMPI<T>::RunImpl() {
  mpi::communicator world;
  int rank = world.rank();
  int size = world.size();

  T local_sum = std::accumulate(local_input_.begin(), local_input_.end(), static_cast<T>(0));
  T global_sum = local_sum;
  int step = 1;

  while (step < size) {
    if (rank % (2 * step) == 0) {
      if (rank + step < size) {
        T recv_value;
        world.recv(rank + step, 0, recv_value);
        global_sum += recv_value;
      }
    } else {
      int target = rank - step;
      world.send(target, 0, global_sum);
      return true;
    }
    step *= 2;
  }

  if (rank == 0) {
    result_ = global_sum;
  }

  for (int proc = 1; proc < size; proc++) {
    world.send(proc, 0, result_);
  }

  if (rank != 0) {
    world.recv(0, 0, result_);
  }

  return true;
}

template <typename T>
bool karaseva_e_reduce_mpi::TestTaskMPI<T>::PostProcessingImpl() {
  mpi::communicator world;
  int rank = world.rank();

  if (rank == 0) {
    if (task_data->outputs[0] == nullptr) {
      task_data->outputs[0] = new uint8_t[sizeof(T)];
    }
    auto* output_ptr = reinterpret_cast<T*>(task_data->outputs[0]);
    *output_ptr = result_;
  }
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