#include "mpi/strakhov_a_m_gauss_jordan/include/ops_mpi.hpp"

#include <boost/mpi/collectives/broadcast.hpp>
#include <cmath>
#include <cstddef>
#include <vector>

namespace {

bool CheckZero(size_t col_size, size_t row_size, std::vector<double>& input) {
  for (size_t i = 0; i < col_size; i++) {
    bool flag1 = true;
    bool flag2 = true;
    for (size_t j = 0; (j < col_size) && (flag1 || flag2); j++) {
      flag1 = flag1 && (input[(j * row_size) + i] == 0);
      flag2 = flag2 && (input[(i * row_size) + j] == 0);
    }
    if (flag1 || flag2) {
      return false;
    }
  }
  return true;
}

void Step(int tkt, size_t i, bool rang_is_head, int dv, std::vector<double>& head_vec,
          std::vector<double>& local_input) {
  size_t row_size = head_vec.size();
  for (size_t k = 0; static_cast<int>(k) < dv; k++) {
    size_t k_row = k * row_size;
    size_t tkt_row = tkt * row_size;
    if (rang_is_head && (tkt == static_cast<int>(k))) {
      if (local_input[tkt_row + i] != 1.0) {
        for (size_t j = i + 1; j < row_size; j++) {
          local_input[tkt_row + j] /= local_input[tkt_row + i];
        }
        local_input[tkt_row + i] = 1.0;
      }
      continue;
    };
    double kf = local_input[k_row + i] / head_vec[i];
    for (size_t j = i + 1; j < row_size; j++) {
      local_input[k_row + j] -= (head_vec[j] * kf);
    }
    local_input[k_row + i] = 0;
  }
}

}  // namespace

bool strakhov_a_m_gauss_jordan_mpi::TestTaskMPI::PreProcessingImpl() {
  // Init value for input and output
  

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
    input_ = std::vector<double>(in_ptr, in_ptr + (row_size_ * col_size_));
    return CheckZero(col_size_, row_size_, input_);
  }
  input_ = std::vector<double>(0);

  return true;
}

bool strakhov_a_m_gauss_jordan_mpi::TestTaskMPI::RunImpl() {
  broadcast(world_, row_size_, 0);
  broadcast(world_, col_size_, 0);
  output_ = std::vector<double>(row_size_, 0);
  int r = world_.rank();
  size_t sz = world_.size();
  size_t ost = col_size_ % sz;
  size_t dv = (col_size_ / sz) + (int)(r < static_cast<int>(ost));
  size_t osn_dv = col_size_ / sz;
  int head = 0;
  int new_head = 0;
  size_t tkt = 0;
  std::vector<double> head_vec(row_size_, 0);
  std::vector<double> local_input;
  if (r == 0) {
    int j = static_cast<int>(dv);
    for (int k = 1; k < world_.size(); k++) {
      size_t pr_size = (k < static_cast<int>(ost)) ? dv : osn_dv;
      world_.send(k, 0, &input_[j * row_size_], static_cast<int>(pr_size * row_size_));
      j += static_cast<int>(pr_size);
    }
    local_input = std::vector<double>(input_.data(), input_.data() + (dv * row_size_));
  } else {
    local_input = std::vector<double>(dv * row_size_, 0);
    world_.recv(0, 0, local_input.data(), static_cast<int>(local_input.size()));
  }
  for (size_t i = 0; i < col_size_; i++) {
    broadcast(world_, tkt, head);
    broadcast(world_, new_head, head);
    head = new_head;
    if (head == r) {
      head_vec = std::vector(local_input.data() + (tkt * row_size_), local_input.data() + ((tkt + 1) * row_size_));
    }
    broadcast(world_, head_vec.data(), static_cast<int>(row_size_), head);
    Step(static_cast<int>(tkt), i, (r == head), static_cast<int>(dv), head_vec, local_input);
    tkt++;
    if (tkt >= dv) {
      new_head++;
      tkt = 0;
    }
  }
  std::vector<double> output_local = std::vector<double>(dv, 0);
  for (size_t i = 0; i < dv; i++) {
    output_local[i] = (float)local_input[(row_size_ * (i + 1)) - 1];
  }
  output_ = std::vector<double>(col_size_, 0);
  if (r != 0) {
    world_.send(0, 0, output_local.data(), static_cast<int>(output_local.size()));
  } else {
    int j = static_cast<int>(dv);
    for (int k = 1; k < world_.size(); k++) {
      int recv_size = static_cast<int>(k < static_cast<int>(ost) ? dv : osn_dv);
      world_.recv(k, 0, &output_[j], recv_size);
      j += recv_size;
    }

    for (int i = 0; i < static_cast<int>(output_local.size()); i++) {
      output_[i] = output_local[i];
    }
  }
  return true;
}

bool strakhov_a_m_gauss_jordan_mpi::TestTaskMPI::PostProcessingImpl() {
  broadcast(world_, output_.data(), static_cast<int>(col_size_), 0);

  if (world_.rank() == 0) {
    for (size_t i = 0; i < output_.size(); i++) {
      reinterpret_cast<double*>(task_data->outputs[0])[i] = output_[i];
    }
  }

  return true;
}
