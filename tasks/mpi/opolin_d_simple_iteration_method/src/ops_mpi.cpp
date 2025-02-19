// Copyright 2024 Nesterov Alexander
#include "mpi/opolin_d_simple_iteration_method/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/mpi/collectives/broadcast.hpp>
#include <boost/mpi/collectives/gatherv.hpp>
#include <boost/mpi/collectives/scatterv.hpp>
#include <boost/serialization/vector.hpp>  // NOLINT(misc-include-cleaner)
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

bool opolin_d_simple_iteration_method_mpi::SimpleIterMethodkMPI::PreProcessingImpl() {
  // init data
  if (world_.rank() == 0) {
    auto *ptr = reinterpret_cast<double *>(task_data->inputs[1]);
    b_.assign(ptr, ptr + n_);
    epsilon_ = *reinterpret_cast<double *>(task_data->inputs[2]);
    C_.resize(n_ * n_, 0.0);
    d_.resize(n_, 0.0);
    Xold_.resize(n_, 0.0);
    Xnew_.resize(n_, 0.0);
    max_iters_ = *reinterpret_cast<int *>(task_data->inputs[3]);
    // generate C matrix and d vector
    for (size_t i = 0; i < n_; ++i) {
      for (size_t j = 0; j < n_; ++j) {
        if (i != j) {
          C_[(i * n_) + j] = -A_[(i * n_) + j] / A_[(i * n_) + i];
        }
      }
      d_[i] = b_[i] / A_[(i * n_) + i];
    }
  }
  return true;
}

bool opolin_d_simple_iteration_method_mpi::SimpleIterMethodkMPI::ValidationImpl() {
  // check input and output
  if (world_.rank() == 0) {
    if (task_data->inputs_count.empty() || task_data->inputs.size() != 4) {
      return false;
    }
    if (task_data->outputs_count.empty() || task_data->inputs_count[0] != task_data->outputs_count[0] ||
        task_data->outputs.empty()) {
      return false;
    }
    n_ = task_data->inputs_count[0];
    if (n_ <= 0) {
      return false;
    }
    auto *ptr = reinterpret_cast<double *>(task_data->inputs[0]);
    A_.assign(ptr, ptr + (n_ * n_));

    // check ranks
    size_t rank_a = Rank(A_, n_);
    if (rank_a != n_) {
      return false;
    }
    // check main diagonal
    for (size_t i = 0; i < n_; ++i) {
      if (std::abs(A_[(i * n_) + i]) < std::numeric_limits<double>::epsilon()) {
        return false;
      }
    }
    if (!IsDiagonalDominance(A_, n_)) {
      return false;
    }
  }
  return true;
}

bool opolin_d_simple_iteration_method_mpi::SimpleIterMethodkMPI::RunImpl() {
  broadcast(world_, n_, 0);
  broadcast(world_, epsilon_, 0);
  broadcast(world_, max_iters_, 0);
  Xnew_.resize(n_);
  Xold_.resize(n_);

  auto base_rows = static_cast<int32_t>(n_ / world_.size());
  auto remainder = static_cast<int32_t>(n_ % world_.size());

  std::vector<int32_t> rows_per_worker(world_.size());
  std::vector<int32_t> elements_per_worker(world_.size());
  for (int rank = 0; rank < world_.size(); ++rank) {
    rows_per_worker[rank] = base_rows + (rank < remainder ? 1 : 0);
    elements_per_worker[rank] = rows_per_worker[rank] * static_cast<int32_t>(n_);
  }

  std::vector<double> local_c(elements_per_worker[world_.rank()]);
  std::vector<double> local_d(rows_per_worker[world_.rank()]);
  std::vector<double> local_x(rows_per_worker[world_.rank()]);

  scatterv(world_, C_, elements_per_worker, local_c.data(), 0);
  scatterv(world_, d_, rows_per_worker, local_d.data(), 0);

  double global_error = 0.0;
  int iteration = 0;
  do {
    broadcast(world_, Xold_, 0);

    for (int i = 0; i < rows_per_worker[world_.rank()]; ++i) {
      double sum = local_d[i];
      for (size_t j = 0; j < Xold_.size(); ++j) {
        sum += local_c[(i * n_) + j] * Xold_[j];
      }
      local_x[i] = sum;
    }

    gatherv(world_, local_x, Xnew_.data(), rows_per_worker, 0);

    if (world_.rank() == 0) {
      global_error = 0.0;
      for (size_t i = 0; i < n_; ++i) {
        double error = std::abs(Xnew_[i] - Xold_[i]);
        global_error = std::max(global_error, error);
      }
    }

    broadcast(world_, global_error, 0);
    if (world_.rank() == 0) {
      Xold_ = Xnew_;
    }
    ++iteration;
    broadcast(world_, iteration, 0);
  } while (iteration < max_iters_ && global_error > epsilon_);
  return true;
}

bool opolin_d_simple_iteration_method_mpi::SimpleIterMethodkMPI::PostProcessingImpl() {
  if (world_.rank() == 0) {
    auto *out = reinterpret_cast<double *>(task_data->outputs[0]);
    std::ranges::copy(Xnew_, out);
  }
  return true;
}

size_t opolin_d_simple_iteration_method_mpi::Rank(std::vector<double> matrix, size_t n) {
  size_t row_count = n;
  if (row_count == 0) {
    return 0;
  }
  size_t col_count = n;
  size_t rank = 0;
  for (size_t col = 0, row = 0; col < col_count && row < row_count; ++col) {
    size_t max_row_idx = row;
    double max_value = std::abs(matrix[(row * n) + col]);
    for (size_t i = row + 1; i < row_count; ++i) {
      double current_value = std::abs(matrix[(i * n) + col]);
      if (current_value > max_value) {
        max_value = current_value;
        max_row_idx = i;
      }
    }
    if (max_value < 1e-10 || std::abs(matrix[(max_row_idx * n) + col]) < 1e-10) {
      continue;
    }

    if (max_row_idx != row) {
      auto begin1 = matrix.begin() + static_cast<std::vector<double>::difference_type>(row * n);
      auto end1 = begin1 + static_cast<std::vector<double>::difference_type>(col_count);
      auto begin2 = matrix.begin() + static_cast<std::vector<double>::difference_type>(max_row_idx * n);
      std::swap_ranges(begin1, end1, begin2);
    }

    double pivot = matrix[(row * n) + col];
    for (size_t j = col; j < col_count; ++j) {
      matrix[(row * n) + j] /= pivot;
    }

    for (size_t i = 0; i < row_count; ++i) {
      if (i == row) {
        continue;
      }
      double factor = matrix[(i * n) + col];
      for (size_t j = col; j < col_count; ++j) {
        matrix[(i * n) + j] -= factor * matrix[(row * n) + j];
      }
    }
    ++rank;
    ++row;
  }
  return rank;
}

bool opolin_d_simple_iteration_method_mpi::IsDiagonalDominance(std::vector<double> mat, size_t dim) {
  for (size_t i = 0; i < dim; i++) {
    double diagonal_value = std::abs(mat[(i * dim) + i]);
    double row_sum = 0.0;

    for (size_t j = 0; j < dim; j++) {
      if (j != i) {
        row_sum += std::abs(mat[(i * dim) + j]);
      }
    }

    if (diagonal_value <= row_sum) {
      return false;
    }
  }
  return true;
}