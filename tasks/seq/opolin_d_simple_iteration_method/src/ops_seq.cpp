// Copyright 2024 Nesterov Alexander
#include "seq/opolin_d_simple_iteration_method/include/ops_seq.hpp"

#include <cmath>
#include <utility>

using namespace std::chrono_literals;

bool opolin_d_simple_iteration_method_seq::TestTaskSequential::PreProcessingImpl() {
  // init data
  auto *ptr = reinterpret_cast<double *>(task_data->inputs[1]);
  b_.assign(ptr, ptr + n_);
  epsilon_ = *reinterpret_cast<double *>(task_data->inputs[2]);
  C_.resize(n_ * n_, 0.0);
  d_.resize(n_, 0.0);
  Xold_.resize(n_, 0.0);
  Xnew_.resize(n_, 0.0);
  max_iter_ = *reinterpret_cast<int *>(task_data->inputs[3]);
  // generate C matrix and d vector
  for (size_t i = 0; i < n_; ++i) {
    for (size_t j = 0; j < n_; ++j) {
      if (i != j) {
        C_[i * n_ + j] = -A_[i * n_ + j] / A_[i * n_ + i];
      }
    }
    d_[i] = b_[i] / A_[i * n_ + i];
  }
  return true;
}

bool opolin_d_simple_iteration_method_seq::TestTaskSequential::ValidationImpl() {
  // check input and output
  if (task_data->inputs_count.empty() || task_data->inputs.size() != 4) return false;
  if (task_data->outputs_count.empty() || task_data->inputs_count[0] != task_data->outputs_count[0] ||
      task_data->outputs.empty())
    return false;

  n_ = task_data->inputs_count[0];
  if (n_ <= 0) return false;
  auto *ptr = reinterpret_cast<double *>(task_data->inputs[0]);
  A_.assign(ptr, ptr + n_ * n_);
  // check ranks
  size_t rankA = Rank(A_, n_);
  if (rankA != n_) return false;

  // check main diagonal
  for (size_t i = 0; i < n_; ++i) {
    if (std::abs(A_[i * n_ + i]) < std::numeric_limits<double>::epsilon()) {
      return false;
    }
  }
  if (!IsDiagonalDominance(A_, n_)) {
    return false;
  }
  return true;
}

bool opolin_d_simple_iteration_method_seq::TestTaskSequential::RunImpl() {
  // simple iteration method
  int iteration = 0;
  while (iteration < max_iter_) {
    for (size_t i = 0; i < n_; ++i) {
      double sum = d_[i];
      for (size_t j = 0; j < n_; ++j) {
        if (i != j) {
          sum += C_[i * n_ + j] * Xold_[j];
        }
      }
      Xnew_[i] = sum;
    }
    double max_error = 0.0;
    for (size_t i = 0; i < n_; ++i) {
      double error = std::abs(Xnew_[i] - Xold_[i]);
      if (error > max_error) {
        max_error = error;
      }
    }
    Xold_ = Xnew_;
    if (max_error < epsilon_) {
      break;
    }
    ++iteration;
  }
  if (iteration == max_iter_) {
    return false;
  }

  return true;
}

bool opolin_d_simple_iteration_method_seq::TestTaskSequential::PostProcessingImpl() {
  auto *out = reinterpret_cast<double *>(task_data->outputs[0]);
  std::copy(Xnew_.begin(), Xnew_.end(), out);
  return true;
}

size_t opolin_d_simple_iteration_method_seq::Rank(std::vector<double> matrix, size_t n) {
  size_t rowCount = n;
  if (rowCount == 0) return 0;
  size_t colCount = n;
  size_t rank = 0;
  for (size_t col = 0, row = 0; col < colCount && row < rowCount; ++col) {
    size_t maxRowIdx = row;
    double maxValue = std::abs(matrix[row * n + col]);
    for (size_t i = row + 1; i < rowCount; ++i) {
      double currentValue = std::abs(matrix[i * n + col]);
      if (currentValue > maxValue) {
        maxValue = currentValue;
        maxRowIdx = i;
      }
    }
    if (maxValue < 1e-10) continue;

    if (maxRowIdx != row) {
      for (size_t j = 0; j < colCount; ++j) {
        double temp = matrix[row * n + j];
        matrix[row * n + j] = matrix[maxRowIdx * n + j];
        matrix[maxRowIdx * n + j] = temp;
      }
    }

    double leadElement = matrix[row * n + col];
    if (std::abs(leadElement) < 1e-10) {
      continue;
    }
    for (size_t j = col; j < colCount; ++j) {
      matrix[row * n + j] /= leadElement;
    }

    for (size_t i = 0; i < rowCount; ++i) {
      if (i != row) {
        double factor = matrix[i * n + col];
        for (size_t j = col; j < colCount; ++j) {
          matrix[i * n + j] -= factor * matrix[row * n + j];
        }
      }
    }
    ++rank;
    ++row;
    if (rank == n) {
      break;
    }
  }
  return rank;
}

bool opolin_d_simple_iteration_method_seq::IsDiagonalDominance(std::vector<double> mat, size_t dim) {
  for (size_t i = 0; i < dim; i++) {
    double diagonal_value = std::abs(mat[i * dim + i]);
    double row_sum = 0.0;

    for (size_t j = 0; j < dim; j++) {
      if (j != i) {
        row_sum += std::abs(mat[i * dim + j]);
      }
    }

    if (diagonal_value <= row_sum) {
      return false;
    }
  }
  return true;
}