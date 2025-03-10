#include "mpi/Konstantinov_I_Gauss_Jordan_method/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/mpi/collectives/broadcast.hpp>
#include <boost/mpi/collectives/gatherv.hpp>
#include <boost/mpi/collectives/scatterv.hpp>
#include <boost/serialization/vector.hpp>
#include <cmath>
#include <cstddef>
#include <vector>

using namespace std::chrono;

void konstantinov_i_gauss_jordan_method_mpi::FindMaxRowAndSwap(int k, int n, std::vector<double>& matrix) {
  int max_row = k;
  for (int i = k + 1; i < n; ++i) {
    if (std::abs(matrix[(i * (n + 1)) + k]) > std::abs(matrix[(max_row * (n + 1)) + k])) {
      max_row = i;
    }
  }
  if (max_row != k) {
    for (int j = k; j <= n; ++j) {
      std::swap(matrix[(k * (n + 1)) + j], matrix[(max_row * (n + 1)) + j]);
    }
  }
}

void konstantinov_i_gauss_jordan_method_mpi::NormalizeRow(int k, int n, std::vector<double>& matrix) {
  double diag = matrix[(k * (n + 1)) + k];
  for (int j = k; j <= n; ++j) {
    matrix[(k * (n + 1)) + j] /= diag;
  }
}

void konstantinov_i_gauss_jordan_method_mpi::ProcessLocalMatrix(size_t local_size, int k, int n,
                                                                std::vector<double>& local_matrix,
                                                                const std::vector<double>& header) {
  for (size_t i = 0; i < (local_size / (n + 1)); ++i) {
    double factor = local_matrix[(i * (n + 1)) + k];
    for (int j = k; j <= n; ++j) {
      local_matrix[(i * (n + 1)) + j] -= header[j] * factor;
    }
  }
}

void konstantinov_i_gauss_jordan_method_mpi::ProcessGaussStep(int k, int n, std::vector<double>& matrix,
                                                              std::vector<double>& header,
                                                              std::vector<int>& send_counts,
                                                              std::vector<int>& displacements,
                                                              boost::mpi::communicator& world,
                                                              std::vector<double>& local_matrix, bool is_forward) {
  if (world.rank() == 0) {
    int offset = is_forward ? (n + 1) * (k + 1) : (n + 1) * k;
    size_t remainder_size = is_forward ? matrix.size() - offset : offset;
    int elements_per_process = static_cast<int>(((remainder_size / (n + 1)) / world.size()) * (n + 1));
    int remainder = static_cast<int>(((remainder_size / (n + 1)) % world.size()) * (n + 1));

    send_counts = std::vector<int>(world.size(), elements_per_process);
    for (int i = 0; i < remainder / (n + 1); i++) {
      send_counts[i] += (n + 1);
    }

    displacements = std::vector<int>(world.size(), is_forward ? offset : 0);
    for (int i = 1; i < world.size(); ++i) {
      displacements[i] = displacements[i - 1] + send_counts[i - 1];
    }
  }

  boost::mpi::broadcast(world, header, 0);
  boost::mpi::broadcast(world, send_counts, 0);
  boost::mpi::broadcast(world, displacements, 0);

  local_matrix.resize(send_counts[world.rank()]);
  boost::mpi::scatterv(world, matrix, send_counts, displacements, local_matrix.data(), send_counts[world.rank()], 0);

  konstantinov_i_gauss_jordan_method_mpi::ProcessLocalMatrix(local_matrix.size(), k, n, local_matrix, header);

  boost::mpi::gatherv(world, local_matrix, matrix.data(), send_counts, displacements, 0);
}

bool konstantinov_i_gauss_jordan_method_mpi::GaussJordanMethodSeq::PreProcessingImpl() {
  n_ = *reinterpret_cast<int*>(task_data->inputs[0]);
  matrix_ = std::vector<double>(reinterpret_cast<double*>(task_data->inputs[1]),
                                reinterpret_cast<double*>(task_data->inputs[1]) + (n_ * (n_ + 1)));
  solution_ = std::vector<double>(n_, 0.0);
  return true;
}

bool konstantinov_i_gauss_jordan_method_mpi::GaussJordanMethodSeq::ValidationImpl() {
  int num_rows = static_cast<int>(task_data->inputs_count[0]);
  int num_cols = (task_data->inputs_count[0] > 0) ? (num_rows + 1) : 0;
  if (num_rows <= 0 || num_cols <= 0) {
    return false;
  }
  auto expected_size = static_cast<size_t>(num_rows) * static_cast<size_t>(num_cols);
  if (task_data->inputs_count[1] != expected_size) {
    return false;
  }
  auto* matrix_data = reinterpret_cast<double*>(task_data->inputs[1]);
  for (int i = 0; i < num_rows; ++i) {
    auto value = matrix_data[(i * num_cols) + i];
    if (value == 0.0) {
      return false;
    }
  }
  return true;
}

bool konstantinov_i_gauss_jordan_method_mpi::GaussJordanMethodSeq::RunImpl() {
  for (int k = 0; k < n_; ++k) {
    int max_row = k;
    for (int i = k + 1; i < n_; ++i) {
      if (std::abs(matrix_[(i * (n_ + 1)) + k]) > std::abs(matrix_[(max_row * (n_ + 1)) + k])) {
        max_row = i;
      }
    }
    if (max_row != k) {
      for (int j = k; j <= n_; ++j) {
        std::swap(matrix_[(k * (n_ + 1)) + j], matrix_[(max_row * (n_ + 1)) + j]);
      }
    }
    double diag = matrix_[(k * (n_ + 1)) + k];
    for (int j = k; j <= n_; ++j) {
      matrix_[(k * (n_ + 1)) + j] /= diag;
    }
    for (int i = k + 1; i < n_; ++i) {
      double factor = matrix_[(i * (n_ + 1)) + k];
      for (int j = k; j <= n_; ++j) {
        matrix_[(i * (n_ + 1)) + j] -= matrix_[(k * (n_ + 1)) + j] * factor;
      }
    }
  }
  for (int k = n_ - 1; k >= 0; --k) {
    for (int i = k - 1; i >= 0; --i) {
      double factor = matrix_[(i * (n_ + 1)) + k];
      for (int j = k; j <= n_; ++j) {
        matrix_[(i * (n_ + 1)) + j] -= matrix_[(k * (n_ + 1)) + j] * factor;
      }
    }
  }
  for (int i = 0; i < n_; ++i) {
    solution_[i] = matrix_[(i * (n_ + 1)) + n_];
  }

  return true;
}

bool konstantinov_i_gauss_jordan_method_mpi::GaussJordanMethodSeq::PostProcessingImpl() {
  for (int i = 0; i < n_; ++i) {
    reinterpret_cast<double*>(task_data->outputs[0])[i] = solution_[i];
  }
  return true;
}

bool konstantinov_i_gauss_jordan_method_mpi::GaussJordanMethodMpi::PreProcessingImpl() {
  if (world_.rank() == 0) {
    n_ = *reinterpret_cast<int*>(task_data->inputs[0]);
    int num_elements = n_ * (n_ + 1);
    solution_ = std::vector<double>();
    matrix_ = std::vector<double>(reinterpret_cast<double*>(task_data->inputs[1]),
                                  reinterpret_cast<double*>(task_data->inputs[1]) + num_elements);
    diag_elements_.resize(n_);
    for (int i = 0; i < n_; ++i) {
      diag_elements_[i] = (i * (n_ + 1) + i);
    }
  }

  return true;
}

bool konstantinov_i_gauss_jordan_method_mpi::GaussJordanMethodMpi::ValidationImpl() {
  if (world_.rank() == 0) {
    int num_rows = static_cast<int>(task_data->inputs_count[0]);
    int num_cols = (task_data->inputs_count[0] > 0) ? (num_rows + 1) : 0;
    if (num_rows <= 0 || num_cols <= 0) {
      return false;
    }
    auto expected_size = static_cast<size_t>(num_rows) * static_cast<size_t>(num_cols);
    if (task_data->inputs_count[1] != expected_size) {
      return false;
    }
    auto* matrix_data = reinterpret_cast<double*>(task_data->inputs[1]);
    for (int i = 0; i < num_rows; ++i) {
      double diag_element = matrix_data[(i * (num_cols)) + i];
      if (std::abs(diag_element) < 1e-9) {
        return false;
      }
    }
  }
  return true;
}

bool konstantinov_i_gauss_jordan_method_mpi::GaussJordanMethodMpi::RunImpl() {
  boost::mpi::broadcast(world_, n_, 0);
  const int row_size = n_ + 1;

  for (int k = 0; k < n_; ++k) {
    if (world_.rank() == 0) {
      konstantinov_i_gauss_jordan_method_mpi::FindMaxRowAndSwap(k, n_, matrix_);
      konstantinov_i_gauss_jordan_method_mpi::NormalizeRow(k, n_, matrix_);
      header_ = std::vector<double>(matrix_.begin() + (k * row_size), matrix_.begin() + (k * row_size) + row_size);
    }
    konstantinov_i_gauss_jordan_method_mpi::ProcessGaussStep(k, n_, matrix_, header_, sendCounts_, displacements_,
                                                             world_, localMatrix_, true);
  }

  for (int k = n_ - 1; k >= 0; --k) {
    if (world_.rank() == 0) {
      header_ = std::vector<double>(matrix_.begin() + (k * row_size), matrix_.begin() + (k * row_size) + row_size);
    }
    konstantinov_i_gauss_jordan_method_mpi::ProcessGaussStep(k, n_, matrix_, header_, sendCounts_, displacements_,
                                                             world_, localMatrix_, false);
  }

  return true;
}

bool konstantinov_i_gauss_jordan_method_mpi::GaussJordanMethodMpi::PostProcessingImpl() {
  world_.barrier();
  if (world_.rank() == 0) {
    for (int i = 0; i < n_; ++i) {
      reinterpret_cast<double*>(task_data->outputs[0])[i] = matrix_[(i * (n_ + 1)) + n_];
    }
  }
  return true;
}