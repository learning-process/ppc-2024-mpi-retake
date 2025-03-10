#include "seq/sedova_o_mult_matrices_ccs/include/ops_seq.hpp"

#include <thread>

void sedova_o_test_task_seq::ConvertToCCS(const std::vector<std::vector<double>>& matrix, std::vector<double>& values,
                                          std::vector<int>& row_indices, std::vector<int>& col_pointers) {
  int num_rows = matrix.size();
  int num_cols = num_rows > 0 ? matrix[0].size() : 0;

  col_pointers.resize(num_cols + 1, 0);

  for (int col = 0; col < num_cols; ++col) {
    for (int row = 0; row < num_rows; ++row) {
      if (matrix[row][col] != 0) {
        col_pointers[col + 1]++;
      }
    }
  }
  for (int col = 0; col < num_cols; ++col) {
    col_pointers[col + 1] += col_pointers[col];
  }
  for (int col = 0; col < num_cols; ++col) {
    for (int row = 0; row < num_rows; ++row) {
      double value = matrix[row][col];
      if (value != 0) {
        values.push_back(value);
        row_indices.push_back(row);
        col_pointers[col]++;
      }
    }
  }
  for (int col = num_cols; col > 0; --col) {
    col_pointers[col] = col_pointers[col - 1];
  }
  col_pointers[0] = 0;
}

template <typename T>
T sedova_o_test_task_seq::MultVectors(const std::vector<T>& vector_A, const std::vector<T>& vector_B) {
  double ans = 0;
  for (size_t i = 0; i < vector_A.size(); ++i) {
    ans += vector_A[i] * vector_B[i];
  }
  return ans;
}

std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> sedova_o_test_task_seq::Convertirovanie(
    const std::vector<double>& A, const std::vector<int>& row_ind_A, const std::vector<int>& col_ind_A,
    const std::vector<double>& B, const std::vector<int>& row_ind_B, const std::vector<int>& col_ind_B, int rows_A,
    int cols_A, int rows_B, int cols_B) {
  std::vector<std::vector<double>> matrix_A(rows_A, std::vector<double>(cols_A, 0));
  for (int i = 0; i < rows_A; ++i) {
    std::vector<double> v(col_ind_A.size() - 1, 0);
    for (size_t j = 0; j < col_ind_A.size() - 1; ++j) {
      for (int ind = col_ind_A[j]; ind < col_ind_A[j + 1]; ++ind) {
        if (row_ind_A[ind] == i) {
          v[j] = A[ind];
        }
      }
    }
    matrix_A[i] = v;
  }
  std::vector<std::vector<double>> matrix_B(cols_B, std::vector<double>(rows_B, 0));
  for (size_t i = 0; i < col_ind_B.size() - 1; ++i) {
    std::vector<double> v(rows_B, 0);
    for (int ind = col_ind_B[i]; ind < col_ind_B[i + 1]; ++ind) {
      v[row_ind_B[ind]] = B[ind];
    }
    matrix_B[i] = v;
  }
  return std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>>(matrixA, matrixB);
}
void sedova_o_test_task_seq::FillData(std::shared_ptr<ppc::core::task_data>& task_data, int rows_A, int cols_A,
                                      int rows_B, int cols_B, std::vector<double>& A, std::vector<int>& row_in_A,
                                      std::vector<int>& col_in_A, std::vector<double>& B, std::vector<int>& row_in_B,
                                      std::vector<int>& col_in_B, std::vector<std::vector<double>>& out) {
  task_data->inputs_count.emplace_back(rows_A);
  task_data->inputs_count.emplace_back(cols_A);
  task_data->inputs_count.emplace_back(A.size());
  task_data->inputs_count.emplace_back(row_in_A.size());
  task_data->inputs_count.emplace_back(col_in_A.size());

  task_data->inputs_count.emplace_back(rows_B);
  task_data->inputs_count.emplace_back(cols_B);
  task_data->inputs_count.emplace_back(B.size());
  task_data->inputs_count.emplace_back(row_in_B.size());
  task_data->inputs_count.emplace_back(col_in_B.size());
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(row_in_A.data()));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(col_in_A.data()));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(B.data()));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(row_in_B.data()));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(col_in_B.data()));

  for (size_t i = 0; i < out.size(); ++i) {
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out[i].data()));
  }
  task_data->outputs_count.emplace_back(out.size());
  task_data->outputs_count.emplace_back(out[0].size());
}

bool sedova_o_test_task_seq::TestTaskSequential::PreProcessingImpl() {
  rows_A = task_data->inputs_count[0];
  cols_A = task_data->inputs_count[1];
  size_A = task_data->inputs_count[2];

  row_insize_A = task_data->inputs_count[3];
  col_insize_A = task_data->inputs_count[4];
  rows_B = task_data->inputs_count[5];
  cols_B = task_data->inputs_count[6];
  size_B = task_data->inputs_count[7];
  row_insize_B = task_data->inputs_count[8];
  col_insize_B = task_data->inputs_count[9];
  A.resize(size_A);
  for (int i = 0; i < size_A; ++i) {
    auto* A_ptr = reinterpret_cast<double*>(task_data->inputs[0]);
    A[i] = A_ptr[i];
  }
  row_in_A.resize(row_insize_A);
  for (int i = 0; i < row_insize_A; ++i) {
    int* row_in_A_ptr = reinterpret_cast<int*>(task_data->inputs[1]);
    row_in_A[i] = row_in_A_ptr[i];
  }
  col_in_A.resize(col_insize_A);
  for (int i = 0; i < col_insize_A; ++i) {
    int* col_in_A_ptr = reinterpret_cast<int*>(task_data->inputs[2]);
    col_in_A[i] = col_in_A_ptr[i];
  }

  B.resize(B_size);
  for (int i = 0; i < B_size; ++i) {
    auto* B_ptr = reinterpret_cast<double*>(task_data->inputs[3]);
    B[i] = B_ptr[i];
  }
  row_in_B.resize(row_in_B_size);
  for (int i = 0; i < row_in_B_size; ++i) {
    int* row_in_B_ptr = reinterpret_cast<int*>(task_data->inputs[4]);
    row_in_B[i] = row_in_B_ptr[i];
  }
  col_in_B.resize(col_in_B_size);
  for (int i = 0; i < col_in_B_size; ++i) {
    int* col_in_B_ptr = reinterpret_cast<int*>(task_data->inputs[5]);
    col_in_B[i] = col_in_B_ptr[i];
  }
  return true;
}

bool sedova_o_test_task_seq::TestTaskSequential::ValidationImpl() {
  int r_A = task_data->inputs_count[0];
  int c_A = task_data->inputs_count[1];
  int r_B = task_data->inputs_count[5];
  int c_B = task_data->inputs_count[6];
  int rows_Ans = task_data->outputs_count[0];
  int cols_Ans = task_data->outputs_count[1];
  return c_A == r_B && cols_Ans == c_B && rows_Ans == r_A;
}

bool sedova_o_test_task_seq::TestTaskSequential::RunImpl() {
  auto pairMatrix = sedova_o_test_task_seq::Convertirovanie(A, row_in_A, col_in_A, B, row_in_B, col_in_B, rows_A,
                                                            cols_A, rows_B, cols_B);
  ans.resize(rows_A, std::vector<double>(cols_B, 0));
  for (int i = 0; i < rows_A; ++i) {
    for (int j = 0; j < cols_B; ++j) {
      ans[i][j] = sedova_o_test_task_seq::MultVectors(pairMatrix.first[i], pairMatrix.second[j]);
    }
  }
  return true;
}

bool sedova_o_test_task_seq::TestTaskSequential::PostProcessingImpl() {
  for (size_t i = 0; i < ans.size(); ++i) {
    for (size_t j = 0; j < ans[i].size(); ++j) {
      reinterpret_cast<double*>(task_data->outputs[i])[j] = ans[i][j];
    }
  }
  return true;
}
