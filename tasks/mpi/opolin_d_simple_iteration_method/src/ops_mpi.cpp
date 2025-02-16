// Copyright 2024 Nesterov Alexander
#include "mpi/opolin_d_simple_iteration_method/include/ops_mpi.hpp"

#include <climits>
#include <cmath>
#include <random>
#include <utility>
#include <vector>

bool opolin_d_simple_iteration_method_mpi::TestTaskMPI::PreProcessingImpl() {
  InternalOrderTest();
  // init data
  if (world_.rank() == 0) {
    auto* ptr = reinterpret_cast<double*>(task_data->inputs[1]);
    b_.assign(ptr, ptr + n_);
    epsilon_ = *reinterpret_cast<double*>(task_data->inputs[2]);
    C_.resize(n_ * n_, 0.0);
    d_.resize(n_, 0.0);
    Xold_.resize(n_, 0.0);
    Xnew_.resize(n_, 0.0);
    max_iters_ = *reinterpret_cast<int*>(task_data->inputs[3]);
    std::vector<double> augmen_matrix = A_;
    for (size_t i = 0; i < n_; ++i) {
      augmen_matrix.push_back(b_[i]);
    }
    // generate C matrix and d vector
    for (size_t i = 0; i < n_; ++i) {
      for (size_t j = 0; j < n_; ++j) {
        if (i != j) {
          C_[i * n_ + j] = -A_[i * n_ + j] / A_[i * n_ + i];
        }
      }
      d_[i] = b_[i] / A_[i * n_ + i];
    }
  }
  return true;
}

bool opolin_d_simple_iteration_method_mpi::TestTaskMPI::ValidationImpl() {
  InternalOrderTest();
  // check input and output
  if (world_.rank() == 0) {
    if (task_data->inputs_count.empty() || task_data->inputs.size() != 4) return false;
    if (task_data->outputs_count.empty() || task_data->inputs_count[0] != task_data->outputs_count[0] ||
        task_data->outputs.empty())
      return false;

    n_ = task_data->inputs_count[0];
    if (n_ <= 0) return false;
    auto* ptr = reinterpret_cast<double*>(task_data->inputs[0]);
    A_.assign(ptr, ptr + n_ * n_);

    // check ranks
    size_t rankA = rank(A_, n_);
    if (rankA != n_) {
      return false;
    }
    // check main diagonal
    for (size_t i = 0; i < n_; ++i) {
      if (std::abs(A_[i * n_ + i]) < std::numeric_limits<double>::epsilon()) {
        return false;
      }
    }
    if (!isDiagonalDominance(A_, n_)) {
      return false;
    }
  }
  return true;
}

bool opolin_d_simple_iteration_method_mpi::TestTaskMPI::RunImpl() {
  InternalOrderTest();

  broadcast(world_, n_, 0);
  broadcast(world_, epsilon_, 0);
  broadcast(world_, max_iters_, 0);
  Xnew_.resize(n_);
  Xold_.resize(n_);

  int32_t base_rows = n_ / world_.size();
  int32_t remainder = n_ % world_.size();

  std::vector<int32_t> rows_per_worker(world_.size());
  std::vector<int32_t> elements_per_worker(world_.size());
  for (int rank = 0; rank < world_.size(); ++rank) {
    rows_per_worker[rank] = base_rows + (rank < remainder ? 1 : 0);
    elements_per_worker[rank] = rows_per_worker[rank] * n_;
  }

  std::vector<double> local_C(elements_per_worker[world_.rank()]);
  std::vector<double> local_d(rows_per_worker[world_.rank()]);
  std::vector<double> local_X(rows_per_worker[world_.rank()]);

  scatterv(world_, C_, elements_per_worker, local_C.data(), 0);
  scatterv(world_, d_, rows_per_worker, local_d.data(), 0);

  double global_error = 0.0;
  int iteration = 0;
  do {
    broadcast(world_, Xold_, 0);

    for (int i = 0; i < rows_per_worker[world_.rank()]; ++i) {
      double sum = local_d[i];
      for (size_t j = 0; j < Xold_.size(); ++j) {
        sum += local_C[i * n_ + j] * Xold_[j];
      }
      local_X[i] = sum;
    }

    gatherv(world_, local_X, Xnew_.data(), rows_per_worker, 0);

    if (world_.rank() == 0) {
      global_error = 0.0;
      for (size_t i = 0; i < n_; ++i) {
        double error = std::abs(Xnew_[i] - Xold_[i]);
        global_error = std::max(global_error, error);
      }
    }

    broadcast(world_, global_error, 0);
    if (world_.rank() == 0) Xold_ = Xnew_;
    ++iteration;
    broadcast(world_, iteration, 0);
  } while (iteration < max_iters_ && global_error > epsilon_);
  return true;
}

bool opolin_d_simple_iteration_method_mpi::TestTaskMPI::PostProcessingImpl() {
  InternalOrderTest();
  if (world_.rank() == 0) {
    for (size_t i = 0; i < Xnew_.size(); i++) {
      reinterpret_cast<int*>(task_data->outputs[0])[i] = Xnew_[i];
    }
  }
  return true;
}

size_t opolin_d_simple_iteration_method_mpi::rank(std::vector<double> matrix, size_t n) {
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

bool opolin_d_simple_iteration_method_mpi::isDiagonalDominance(std::vector<double> mat, size_t dim) {
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

void opolin_d_simple_iteration_method_mpi::generateTestData(size_t size, std::vector<double> &X, std::vector<double> &A, std::vector<double> &b) {
  std::srand(static_cast<unsigned>(std::time(nullptr)));

  X.resize(size);
  for (size_t i = 0; i < size; ++i) {
    X[i] = -10.0 + static_cast<double>(std::rand() % 1000) / 50.0;
  }

  A.resize(size * size, 0.0);
  for (size_t i = 0; i < size; ++i) {
    double sum = 0.0;
    for (size_t j = 0; j < size; ++j) {
      if (i != j) {
        A[i * size + j] = -1.0 + static_cast<double>(std::rand() % 1000) / 500.0;
        sum += std::abs(A[i * size + j]);
      }
    }
    A[i * size + i] = sum + 1.0;
  }
  b.resize(size, 0.0);
  for (size_t i = 0; i < size; ++i) {
    for (size_t j = 0; j < size; ++j) {
      b[i] += A[i * size + j] * X[j];
    }
  }
}
