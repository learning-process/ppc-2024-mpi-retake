#include "mpi/strakhov_a_m_gauss_jordan/include/ops_mpi.hpp"

#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

bool strakhov_a_m_gauss_jordan_mpi::TestTaskMPI::PreProcessingImpl() {
  // Init value for input and output
  output_ = std::vector<double>(row_size_, 0);
  return true;
}

bool strakhov_a_m_gauss_jordan_mpi::TestTaskMPI::ValidationImpl() {
  // Check equality of counts elements
  row_size_ = 0;
  col_size_ = 0;
  if (world_.rank() == 0) {
    row_size_ = task_data->inputs_count[0];
    col_size_ = task_data->inputs_count[1];
    if (task_data->inputs_count[1] == 0) {
      return false;
    }
    if (task_data->inputs_count[0] != (task_data->inputs_count[1] + 1)) {
      return false;
    }
    if (task_data->inputs_count[1] != task_data->outputs_count[0]) {
      return false;
    }
    auto* in_ptr = reinterpret_cast<double*>(task_data->inputs[0]);
    input_ = std::vector<double>(in_ptr, in_ptr + row_size_ * col_size_);
    for (size_t i = 0; i < col_size_; i++) {
      bool flag1 = true;
      bool flag2 = true;
      for (size_t j = 0; j < col_size_; j++) {
        if (flag1 && (input_[j * row_size_ + i] != 0)) {
          flag1 = false;
          if (!flag2) {
            break;
          }
        }
        if (flag2 && (input_[i * row_size_ + j] != 0)) {
          flag2 = false;
          if (!flag1) {
            break;
          }
        }
      }
      if (flag1 || flag2) {
        return false;
      }
    }
  } else {
    input_ = std::vector<double>(0);
  }
  broadcast(world_, row_size_, 0);
  broadcast(world_, col_size_, 0);
  return true;
}

bool strakhov_a_m_gauss_jordan_mpi::TestTaskMPI::RunImpl() {
  int r = world_.rank();
  size_t sz = world_.size();
  size_t ost = col_size_ % sz;
  size_t dv = col_size_ / sz + (int)(r < ost);
  size_t osn_dv = col_size_ / sz;
  int head = 0;
  size_t tkt = 0;
  std::vector<double> head_vec(row_size_, 0);
  std::vector<double> local_input;
  if (r == 0) {
    int j = dv;
    for (int k = 1; k < world_.size(); k++) {
      size_t pr_size = (k < ost) ? dv : osn_dv;
      world_.send(k, 0, &input_[j * row_size_], pr_size * row_size_);
      j += pr_size;
    }
    local_input = std::vector<double>(input_.data(), input_.data() + dv * row_size_);
  } else {
    local_input = std::vector<double>(dv * row_size_, 0);
    world_.recv(0, 0, local_input.data(), local_input.size());
  }

  if (r == 0) {
    for (size_t i = 0; i < col_size_; i++) {
      broadcast(world_, head, 0);
      broadcast(world_, tkt, 0);
      if (head == 0) {
        if (local_input[tkt * row_size_ + i] != 1.0) {
          for (size_t j = i + 1; j < row_size_; j++) {
            local_input[tkt * row_size_ + j] /= local_input[tkt * row_size_ + i];
          }
          local_input[tkt * row_size_ + i] = 1.0;
        }
        head_vec = std::vector(local_input.data() + tkt * row_size_, local_input.data() + (tkt + 1) * row_size_);
      }
      broadcast(world_, head_vec.data(), row_size_, head);

      for (size_t k = 0; k < dv; k++) {
        if ((head == 0) && (tkt == k)) {
          continue;
        }
        double kf = local_input[k * row_size_ + i] / head_vec[i];
        for (size_t j = i + 1; j < row_size_; j++) {
          local_input[k * row_size_ + j] -= (head_vec[j] * kf);
        }
        local_input[k * row_size_ + i] = 0;
      }
      tkt++;

      if ((head >= ost) && (tkt >= osn_dv)) {
        head++;
        tkt = 0;
        continue;
      }
      if ((head < ost) && (tkt >= dv)) {
        head++;
        tkt = 0;
      }
    }
  } else {
    for (size_t i = 0; i < col_size_; i++) {
      broadcast(world_, head, 0);
      broadcast(world_, tkt, 0);

      if (head == r) {
        head_vec = std::vector(local_input.data() + tkt * row_size_, local_input.data() + (tkt + 1) * row_size_);
      }

      broadcast(world_, head_vec.data(), row_size_, head);

      for (size_t k = 0; k < dv; k++) {
        if ((head == r) && (tkt == k)) {
          if (local_input[tkt * row_size_ + i] != 1.0) {
            for (size_t j = i + 1; j < row_size_; j++) {
              local_input[tkt * row_size_ + j] /= local_input[tkt * row_size_ + i];
            }
            local_input[tkt * row_size_ + i] = 1.0;
          }
          continue;
        };
        double kf = local_input[k * row_size_ + i] / head_vec[i];
        for (size_t j = i + 1; j < row_size_; j++) {
          local_input[k * row_size_ + j] -= (head_vec[j] * kf);
        }
        local_input[k * row_size_ + i] = 0;
      }
    }
  }

  std::vector<double> output_local = std::vector<double>(dv, 0);
  for (size_t i = 0; i < dv; i++) {
    output_local[i] = (float)local_input[row_size_ * (i + 1) - 1];
  }

  world_.barrier();

  output_ = std::vector<double>(col_size_, 0);

  if (r != 0) {
    MPI_Send(output_local.data(), output_local.size(), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
  } else {
    int j = dv;
    for (int k = 1; k < world_.size(); k++) {
      int recv_size = (k < ost) ? dv : osn_dv;
      MPI_Recv(&output_[j], recv_size, MPI_DOUBLE, k, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      j += recv_size;
    }

    for (int i = 0; i < output_local.size(); i++) {
      output_[i] = output_local[i];
    }
  }
  world_.barrier();
  return true;
}

bool strakhov_a_m_gauss_jordan_mpi::TestTaskMPI::PostProcessingImpl() {
  broadcast(world_, output_.data(), col_size_, 0);

  if (world_.rank() == 0) {
    for (size_t i = 0; i < output_.size(); i++) {
      reinterpret_cast<double*>(task_data->outputs[0])[i] = output_[i];
    }
  }

  return true;
}
