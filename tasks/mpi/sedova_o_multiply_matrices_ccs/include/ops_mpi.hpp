#pragma once

#include <algorithm>
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
    col_ptr.push_back(values.size());  // NOLINT
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
    t_col_ptr.push_back(t_values.size());  // NOLINT
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
    new_col_ptr.push_back(new_values.size());  // NOLINT
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

  std::vector<int> x(num_rows_a, -1);
  std::vector<double> x_values(num_rows_a, 0.0);

  for (int col_b = 0; col_b < num_cols_b; ++col_b) {
    std::ranges::fill(x.begin(), x.end(), -1);
    std::ranges::fill(x_values.begin(), x_values.end(), 0.0);

    for (int i = col_ptr_b[col_b]; i < col_ptr_b[col_b + 1]; ++i) {
      int row_b = row_indices_b[i];
      x[row_b] = i;
      x_values[row_b] = values_b[i];
    }

    for (int col_a = 0; col_a < static_cast<int>(col_ptr_a.size() - 1); ++col_a) {
      double sum = 0.0;
      for (int i = col_ptr_a[col_a]; i < col_ptr_a[col_a + 1]; ++i) {
        int row_a = row_indices_a[i];
        if (x[row_a] != -1) {
          sum += values_a[i] * x_values[row_a];
        }
      }
      if (sum != 0.0) {
        values_c.push_back(sum);
        row_indices_c.push_back(col_a);
      }
    }

    col_ptr_c.push_back(values_c.size());  //  NOLINT
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
  int rows_A_, cols_A_, rows_B_, cols_B_, rows_At_, cols_At_;
  std::vector<std::vector<double>> A_, B_;
  std::vector<double> A_val_, B_val_, At_val_;
  std::vector<int> A_row_ind_, A_col_ptr_, B_row_ind_, B_col_ptr_, At_row_ind_, At_col_ptr_;
  int color_, loc_start_, loc_end_, loc_cols_;
  std::vector<double> loc_val_, loc_res_val_, res_val_;
  std::vector<int> loc_row_ind_, loc_col_ptr_, loc_res_row_ind_, loc_res_col_ptr_, res_ind_, res_ptr_;

  boost::mpi::communicator world_, comm_;
};

}  // namespace sedova_o_multiply_matrices_ccs_mpi