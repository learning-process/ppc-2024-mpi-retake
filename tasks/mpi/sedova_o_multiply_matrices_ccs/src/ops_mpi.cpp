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

  if (world.rank() == 0) {
  // Загрузка матрицы A
  auto* a_val_ptr = reinterpret_cast<double*>(task_data->inputs[4]);
  A_val.assign(a_val_ptr, a_val_ptr + task_data->inputs_count[4]);

  auto* a_row_ind_ptr = reinterpret_cast<int*>(task_data->inputs[5]);
  A_row_ind.assign(a_row_ind_ptr, a_row_ind_ptr + task_data->inputs_count[5]);

  auto* a_col_ptr_ptr = reinterpret_cast<int*>(task_data->inputs[6]);
  A_col_ptr.assign(a_col_ptr_ptr, a_col_ptr_ptr + task_data->inputs_count[6]);

  // Загрузка матрицы B
  auto* b_val_ptr = reinterpret_cast<double*>(task_data->inputs[7]);
  B_val.assign(b_val_ptr, b_val_ptr + task_data->inputs_count[7]);

  auto* b_row_ind_ptr = reinterpret_cast<int*>(task_data->inputs[8]);
  B_row_ind.assign(b_row_ind_ptr, b_row_ind_ptr + task_data->inputs_count[8]);

  auto* b_col_ptr_ptr = reinterpret_cast<int*>(task_data->inputs[9]);
  B_col_ptr.assign(b_col_ptr_ptr, b_col_ptr_ptr + task_data->inputs_count[9]);

  // Транспонирование матрицы A
  Transponirovanie(A_val, A_row_ind, A_col_ptr, rows_A, cols_A, At_val, At_row_ind, At_col_ptr);

  rows_At = cols_A;
  cols_At = rows_A;
  }
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
  color = static_cast<int>(world.rank() < cols_B);
  comm = world.split(color);

  boost::mpi::broadcast(comm, B_val, 0);
  boost::mpi::broadcast(comm, B_row_ind, 0);
  boost::mpi::broadcast(comm, B_col_ptr, 0);

  boost::mpi::broadcast(comm, At_val, 0);
  boost::mpi::broadcast(comm, At_row_ind, 0);
  boost::mpi::broadcast(comm, At_col_ptr, 0);
  boost::mpi::broadcast(comm, rows_At, 0);

  if (color == 1) {
    auto pair = Segments(cols_B, comm.size(), comm.rank());

    loc_start_ = pair.first;
    loc_end_ = pair.second;
    loc_cols_ = loc_end_ - loc_start_;

    Extract(B_val, B_row_ind, B_col_ptr, loc_start, loc_end, loc_val, loc_row_ind, loc_col_ptr);

    MultiplyCCS(At_val, At_row_ind, At_col_ptr, rows_At, loc_val, loc_row_ind, loc_col_ptr, loc_cols, loc_res_val,
                 loc_res_row_ind, loc_res_col_ptr);
    std::vector<int> sizes_val;
    if (comm.rank() == 0) {
      sizes_val.resize(comm.size());
    }
    boost::mpi::gather(comm, loc_res_col_ptr.back(), sizes_val.data(), 0);

    int size_ptr_vector = 0;
    boost::mpi::reduce(comm, static_cast<int>(loc_res_col_ptr.size() - 1), size_ptr_vector, std::plus<>(), 0);

    std::vector<int> sizes_ptr;
    if (comm.rank() == 0) {
      sizes_ptr.resize(size_ptr_vector);
    }
    boost::mpi::gather(comm, static_cast<int>(loc_res_col_ptr.size() - 1), sizes_ptr, 0);

    if (comm.rank() == 0) {
      int sum = std::accumulate(sizes_val.begin(), sizes_val.end(), 0);
      res_val.resize(sum);
      res_ind.resize(sum);
      res_ptr.resize(size_ptr_vector);

      boost::mpi::gatherv(comm, loc_res_val.data(), loc_res_val.size(), res_val.data(), sizes_val, 0);
      boost::mpi::gatherv(comm, loc_res_row_ind.data(), loc_res_row_ind.size(), res_ind.data(), sizes_val, 0);
      boost::mpi::gatherv(comm, loc_res_col_ptr.data(), loc_res_col_ptr.size() - 1, res_ptr.data(), sizes_ptr, 0);

      int shift = 0;
      int offset = 0;
      for (size_t j = 0; j < sizes_ptr.size(); j++) {
        shift = sizes_val[j];
        offset += sizes_ptr[j];
        for (size_t i = offset; i < res_ptr.size(); i++) {
          res_ptr[i] += shift;
        }
      }

      res_ptr.push_back(sum);
    } else {
      boost::mpi::gatherv(comm, loc_res_val.data(), loc_res_val.size(), 0);
      boost::mpi::gatherv(comm, loc_res_row_ind.data(), loc_res_row_ind.size(), 0);
      boost::mpi::gatherv(comm, loc_res_col_ptr.data(), loc_res_col_ptr.size() - 1, 0);
    }
  }

  return true;
}

bool sedova_o_multiply_matrices_ccs_mpi::TestTaskMPI::PostProcessingImpl() {
  if (color == 1 && comm.rank() == 0) {
    auto* c_val_ptr = reinterpret_cast<double*>(task_data->outputs[0]);
    auto* c_row_ind_ptr = reinterpret_cast<int*>(task_data->outputs[1]);
    auto* c_col_ptr_ptr = reinterpret_cast<int*>(task_data->outputs[2]);

    std::copy(res_val.begin(), res_val.end(), c_val_ptr);
    std::copy(res_ind.begin(), res_ind.end(), c_row_ind_ptr);
    std::copy(res_ptr.begin(), res_ptr.end(), c_col_ptr_ptr);
  }

  return true;
}
