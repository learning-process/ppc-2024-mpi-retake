#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/serialization/array.hpp>   //NOLINT
#include <boost/serialization/vector.hpp>  //NOLINT
#include <cstddef>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace sedova_o_multiply_matrices_ccs_mpi {

inline void Convertirovanie(const std::vector<std::vector<double>>& matrix, int rows, int cols,
                            std::vector<double>& values, std::vector<int>& row_indices, std::vector<int>& col_ptr) {
  col_ptr.clear();
  col_ptr.push_back(0);
  for (int j = 0; j < cols; ++j) {
    for (int i = 0; i < rows; ++i) {
      if (matrix[i][j] != 0.0) {
        values.push_back(matrix[i][j]);
        row_indices.push_back(i);
      }
    }
    col_ptr.push_back(static_cast<int>(values.size()));
  }
}

inline void Transponirovanie(const std::vector<double>& values, const std::vector<int>& row_indices,
                             const std::vector<int>& col_ptr, int rows, int cols, std::vector<double>& t_values,
                             std::vector<int>& t_row_indices, std::vector<int>& t_col_ptr) {
  std::vector<std::vector<int>> int_vectors(rows);
  std::vector<std::vector<double>> real_vectors(rows);

  for (int col = 0; col < cols; ++col) {
    for (int i = col_ptr[col]; i < col_ptr[col + 1]; ++i) {
      int row = row_indices[i];
      double value = values[i];

      int_vectors[row].push_back(col);
      real_vectors[row].push_back(value);
    }
  }

  t_col_ptr.clear();
  t_values.clear();
  t_row_indices.clear();

  t_col_ptr.push_back(0);
  for (int i = 0; i < rows; ++i) {
    for (size_t j = 0; j < int_vectors[i].size(); ++j) {
      t_row_indices.push_back(int_vectors[i][j]);
      t_values.push_back(real_vectors[i][j]);
    }
    t_col_ptr.(static_cast<int>(t_values.size()));
  }
}

inline void Extract(const std::vector<double>& values, const std::vector<int>& row_indices,
                    const std::vector<int>& col_ptr, int start_col, int end_col, std::vector<double>& new_values,
                    std::vector<int>& new_row_indices, std::vector<int>& new_col_ptr) {
  new_values.clear();
  new_row_indices.clear();
  new_col_ptr.clear();

  new_col_ptr.push_back(0);

  for (int j = start_col; j < end_col; ++j) {
    for (int k = col_ptr[j]; k < col_ptr[j + 1]; ++k) {
      new_values.push_back(values[k]);
      new_row_indices.push_back(row_indices[k]);
    }
    new_col_ptr.(static_cast<int>(new_values.size()));
  }
}

inline std::pair<int, int> Segments(int n, int size, int rank) {
  std::vector<std::pair<int, int>> segments;

  int base_size = n / size;
  int remainder = n % size;

  int start = 0;
  for (int i = 0; i < size; ++i) {
    int end = start + base_size + (i < remainder ? 1 : 0);
    segments.emplace_back(start, end);
    start = end;
  }

  return segments[rank];
}

inline void MultiplyCCS(const std::vector<double>& values_a, const std::vector<int>& row_indices_a,
                        const std::vector<int>& col_ptr_a, int num_rows_a, const std::vector<double>& values_b,
                        const std::vector<int>& row_indices_b, const std::vector<int>& col_ptr_b, int num_cols_b,
                        std::vector<double>& values_c, std::vector<int>& row_indices_c, std::vector<int>& col_ptr_c) {
  values_c.clear();
  row_indices_c.clear();
  col_ptr_c.clear();

  col_ptr_c.clear();
  col_ptr_c.push_back(0);

  std::vector<int> X(num_rows_a, -1);
  std::vector<double> X_values(num_rows_a, 0.0);

  for (int col_B = 0; col_B < num_cols_b; ++col_B) {
    std::fill(X.begin(), X.end(), -1);
    std::fill(X_values.begin(), X_values.end(), 0.0);

    for (int i = col_ptr_b[col_B]; i < col_ptr_b[col_B + 1]; ++i) {
      int row_B = row_indices_b[i];
      X[row_B] = i;
      X_values[row_B] = values_b[i];
    }

    for (int col_A = 0; col_A < static_cast<int>(col_ptr_a.size() - 1); ++col_A) {
      double sum = 0.0;
      for (int i = col_ptr_a[col_A]; i < col_ptr_a[col_A + 1]; ++i) {
        int row_A = row_indices_a[i];
        if (X[row_A] != -1) {
          sum += values_a[i] * X_values[row_A];
        }
      }
      if (sum != 0.0) {
        values_c.push_back(sum);
        row_indices_c.push_back(col_A);
      }
    }

    col_ptr_c.push_back(values_c.size());
  }
}

class TestTaskMPI : public ppc::core::Task {
 public:
  explicit TestTaskMPI(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  int rows_A, cols_A, rows_B, cols_B, rows_At, cols_At;
  std::vector<std::vector<double>> A, B;
  std::vector<double> A_val, B_val, At_val;
  std::vector<int> A_row_ind, A_col_ptr, B_row_ind, B_col_ptr, At_row_ind, At_col_ptr;
  int color, loc_start, loc_end, loc_cols;
  std::vector<double> loc_val, loc_res_val, res_val;
  std::vector<int> loc_row_ind, loc_col_ptr, loc_res_row_ind, loc_res_col_ptr, res_ind, res_ptr;

  boost::mpi::communicator world, comm;
};

}  // namespace sedova_o_multiply_matrices_ccs_mpi