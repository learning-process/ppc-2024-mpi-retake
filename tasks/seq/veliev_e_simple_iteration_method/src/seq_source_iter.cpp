// Copyright 2024 Nesterov Alexander
#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#include "seq/veliev_e_simple_iteration_method/include/seq_header_iter.hpp"

namespace veliev_e_simple_iteration_method_seq {

bool VelievSlaeIterSeq::IsDiagonallyDominant() {
  for (int row = 0; row < matrix_size_; ++row) {
    double diag_value = std::abs(MatrixAt(coeff_matrix_, row, row));
    double row_sum = 0.0;

    for (int col = 0; col < matrix_size_; ++col) {
      if (row != col) {
        row_sum += std::abs(MatrixAt(coeff_matrix_, row, col));
      }
    }

    if (diag_value <= row_sum) {
      return false;
    }
  }
  return true;
}

bool VelievSlaeIterSeq::ValidationImpl() {
  if (task_data->inputs_count[0] != task_data->inputs_count[1] ||
      task_data->inputs_count[0] != task_data->outputs_count[0]) {
    return false;
  }
  matrix_size_ = static_cast<int>(task_data->inputs_count[0]);
  coeff_matrix_.resize(matrix_size_ * matrix_size_);
  std::ranges::copy(reinterpret_cast<double*>(task_data->inputs[0]),
                    reinterpret_cast<double*>(task_data->inputs[0]) + (matrix_size_ * matrix_size_),
                    coeff_matrix_.begin());
  rhs_vector_.resize(matrix_size_);
  std::ranges::copy(reinterpret_cast<double*>(task_data->inputs[1]),
                    reinterpret_cast<double*>(task_data->inputs[1]) + matrix_size_, rhs_vector_.begin());
  return IsDiagonallyDominant();
}

bool VelievSlaeIterSeq::PreProcessingImpl() {
  solution_vector_.resize(matrix_size_);
  std::ranges::copy(reinterpret_cast<double*>(task_data->outputs[0]),
                    reinterpret_cast<double*>(task_data->outputs[0]) + matrix_size_, solution_vector_.begin());
  convergence_tolerance_ = 1e-6;

  iteration_matrix_.resize(matrix_size_ * matrix_size_, 0.0);
  free_term_vector_.resize(matrix_size_);
  for (int row = 0; row < matrix_size_; ++row) {
    double diag_value = MatrixAt(coeff_matrix_, row, row);
    if (diag_value == 0.0) {
      return false;
    }
    free_term_vector_[row] = rhs_vector_[row] / diag_value;
    for (int col = 0; col < matrix_size_; ++col) {
      if (row != col) {
        MatrixAt(iteration_matrix_, row, col) = -MatrixAt(coeff_matrix_, row, col) / diag_value;
      }
    }
  }
  return true;
}

bool VelievSlaeIterSeq::RunImpl() {
  std::vector<double> next_solution(matrix_size_, 0.0);
  int iteration = 0;
  while (true) {
    double max_difference = 0.0;
    for (int row = 0; row < matrix_size_; ++row) {
      double sum = 0.0;
      for (int col = 0; col < matrix_size_; ++col) {
        if (row != col) {
          sum += MatrixAt(iteration_matrix_, row, col) * solution_vector_[col];
        }
      }
      next_solution[row] = free_term_vector_[row] + sum;
      max_difference = std::max(max_difference, std::abs(next_solution[row] - solution_vector_[row]));
    }
    if (max_difference <= convergence_tolerance_) {
      break;
    }
    solution_vector_ = next_solution;
    ++iteration;
    if (iteration > 10000) {
      return false;
    }
  }
  return true;
}

bool VelievSlaeIterSeq::PostProcessingImpl() {
  std::ranges::copy(solution_vector_, reinterpret_cast<double*>(task_data->outputs[0]));
  return true;
}

}  // namespace veliev_e_simple_iteration_method_seq
