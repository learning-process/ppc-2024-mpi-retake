#include "mpi/sedova_o_multiply_matrices_ccs/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/mpi/collectives/broadcast.hpp>
#include <boost/mpi/collectives/gather.hpp>
#include <boost/mpi/collectives/reduce.hpp>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <vector>

bool sedova_o_multiply_matrices_ccs_mpi::TestTaskMPI::PreProcessingImpl() {
  rows_A_ = *reinterpret_cast<int*>(task_data->inputs[0]);
  cols_A_ = *reinterpret_cast<int*>(task_data->inputs[1]);
  rows_B_ = *reinterpret_cast<int*>(task_data->inputs[2]);
  cols_B_ = *reinterpret_cast<int*>(task_data->inputs[3]);

  // Загрузка матрицы A
  auto* a_val_ptr = reinterpret_cast<double*>(task_data->inputs[4]);
  A_val_.assign(a_val_ptr, a_val_ptr + task_data->inputs_count[4]);

  auto* a_row_ind_ptr = reinterpret_cast<int*>(task_data->inputs[5]);
  A_row_ind_.assign(a_row_ind_ptr, a_row_ind_ptr + task_data->inputs_count[5]);

  auto* a_col_ptr_ptr = reinterpret_cast<int*>(task_data->inputs[6]);
  A_col_ptr_.assign(a_col_ptr_ptr, a_col_ptr_ptr + task_data->inputs_count[6]);

  // Загрузка матрицы B
  auto* b_val_ptr = reinterpret_cast<double*>(task_data->inputs[7]);
  B_val_.assign(b_val_ptr, b_val_ptr + task_data->inputs_count[7]);

  auto* b_row_ind_ptr = reinterpret_cast<int*>(task_data->inputs[8]);
  B_row_ind_.assign(b_row_ind_ptr, b_row_ind_ptr + task_data->inputs_count[8]);

  auto* b_col_ptr_ptr = reinterpret_cast<int*>(task_data->inputs[9]);
  B_col_ptr_.assign(b_col_ptr_ptr, b_col_ptr_ptr + task_data->inputs_count[9]);

  // Транспонирование матрицы A
  Transponirovanie(A_val_, A_row_ind_, A_col_ptr_, rows_A_, cols_A_, At_val_, At_row_ind_, At_col_ptr_);

  rows_At_ = cols_A_;
  cols_At_ = rows_A_;

  return true;
}

bool sedova_o_multiply_matrices_ccs_mpi::TestTaskMPI::ValidationImpl() {
  int rows_a = *reinterpret_cast<int*>(task_data->inputs[0]);
  int cols_a = *reinterpret_cast<int*>(task_data->inputs[1]);
  int rows_a = *reinterpret_cast<int*>(task_data->inputs[2]);
  int cols_a = *reinterpret_cast<int*>(task_data->inputs[3]);

  return rows_a > 0 && cols_a > 0 && rows_b > 0 && cols_b > 0 && cols_a == rows_b;
}

bool sedova_o_multiply_matrices_ccs_mpi::TestTaskMPI::RunImpl() {
  color_ = static_cast<int>(world_.rank() < cols_B_);
  comm_ = world_.split(color_);

  boost::mpi::broadcast(comm_, B_val_, 0);
  boost::mpi::broadcast(comm_, B_row_ind_, 0);
  boost::mpi::broadcast(comm_, B_col_ptr_, 0);

  boost::mpi::broadcast(comm_, At_val_, 0);
  boost::mpi::broadcast(comm_, At_row_ind_, 0);
  boost::mpi::broadcast(comm_, At_col_ptr_, 0);
  boost::mpi::broadcast(comm_, rows_At_, 0);

  if (color_ == 1) {
    auto pair = Segments(cols_B_, comm_.size(), comm_.rank());

    loc_start_ = pair.first;
    loc_end_ = pair.second;
    loc_cols_ = loc_end_ - loc_start_;

    Extract(B_val_, B_row_ind_, B_col_ptr_, loc_start_, loc_end_, loc_val_, loc_row_ind_, loc_col_ptr_);

    loc_res_val_.clear();
    loc_res_row_ind_.clear();
    loc_res_col_ptr_.clear();

    loc_res_col_ptr_.clear();
    loc_res_col_ptr_.push_back(0);

    std::vector<int> x(rows_At_, -1);
    std::vector<double> x_values(rows_At_, 0.0);

    for (int cols_b = 0; cols_b < loc_cols_; ++cols_b) {
      std::ranges::fill(x.begin(), x.end(), -1);
      std::ranges::fill(x_values.begin(), x_values.end(), 0.0);

      for (int i = loc_col_ptr_[col_b]; i < loc_col_ptr_[col_b + 1]; ++i) {
        int cols_b = loc_row_ind_[i];
        x[cols_b] = i;
        x_values[cols_b] = loc_val_[i];
      }

      for (int col_a = 0; col_a < static_cast<int>(At_col_ptr_.size() - 1); ++col_a) {
        double sum = 0.0;
        for (int i = At_col_ptr_[col_a]; i < At_col_ptr_[col_a + 1]; ++i) {
          int row_a = At_row_ind_[i];
          if (X[row_a] != -1) {
            sum += At_val_[i] * X_values[row_a];
          }
        }
        if (sum != 0.0) {
          loc_res_val_.push_back(sum);
          loc_res_row_ind_.push_back(col_a);
        }
      }

      loc_res_col_ptr_.push_back(loc_res_val_.size());
    }
  }
  std::vector<int> sizes_val;
  if (comm_.rank() == 0) {
    sizes_val.resize(comm_.size());
  }
  boost::mpi::gather(comm_, loc_res_col_ptr_.back(), sizes_val.data(), 0);

  int size_ptr_vector = 0;
  boost::mpi::reduce(comm_, static_cast<int>(loc_res_col_ptr_.size() - 1), size_ptr_vector, std::plus<>(), 0);

  std::vector<int> sizes_ptr;
  if (comm_.rank() == 0) {
    sizes_ptr.resize(size_ptr_vector);
  }
  boost::mpi::gather(comm_, static_cast<int>(loc_res_col_ptr_.size() - 1), sizes_ptr, 0);

  if (comm_.rank() == 0) {
    int sum = std::accumulate(sizes_val.begin(), sizes_val.end(), 0);
    res_val_.resize(sum);
    res_ind_.resize(sum);
    res_ptr_.resize(size_ptr_vector);

    boost::mpi::gatherv(comm_, loc_res_val_.data(), loc_res_val_.size(), res_val_.data(), sizes_val, 0);
    boost::mpi::gatherv(comm_, loc_res_row_ind_.data(), loc_res_row_ind_.size(), res_ind_.data(), sizes_val, 0);
    boost::mpi::gatherv(comm_, loc_res_col_ptr_.data(), loc_res_col_ptr_.size() - 1, res_ptr_.data(), sizes_ptr, 0);

    int shift = 0;
    int offset = 0;
    for (size_t j = 0; j < sizes_ptr.size(); j++) {
      shift = sizes_val[j];
      offset += sizes_ptr[j];
      for (size_t i = offset; i < res_ptr_.size(); i++) {
        res_ptr_[i] += shift;
      }
    }

    res_ptr_.push_back(sum);
  } else {
    boost::mpi::gatherv(comm_, loc_res_val_.data(), loc_res_val_.size(), 0);
    boost::mpi::gatherv(comm_, loc_res_row_ind_.data(), loc_res_row_ind_.size(), 0);
    boost::mpi::gatherv(comm_, loc_res_col_ptr_.data(), loc_res_col_ptr_.size() - 1, 0);
  }
}
return true;
}

bool sedova_o_multiply_matrices_ccs_mpi::TestTaskMPI::PostProcessingImpl() {
  if (color_ == 1 && comm_.rank() == 0) {
    auto* C_val_ptr = reinterpret_cast<double*>(task_data->outputs[0]);
    auto* C_row_ind_ptr = reinterpret_cast<int*>(task_data->outputs[1]);
    auto* C_col_ptr_ptr = reinterpret_cast<int*>(task_data->outputs[2]);

    std::copy(res_val_.begin(), res_val_.end(), C_val_ptr);
    std::copy(res_ind_.begin(), res_ind_.end(), C_row_ind_ptr);
    std::copy(res_ptr_.begin(), res_ptr_.end(), C_col_ptr_ptr);
  }

  return true;
}
