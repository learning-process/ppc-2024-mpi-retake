#include "mpi/Konstantinov_I_Gauss_Jordan_method/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/mpi/collectives/broadcast.hpp>
#include <boost/mpi/collectives/gather.hpp>
#include <boost/mpi/collectives/scatter.hpp>
#include <boost/serialization/vector.hpp>
#include <cmath>
#include <cstddef>
#include <vector>

using namespace std::chrono;

void konstantinov_i_gauss_jordan_method_mpi::FindMaxRowAndSwap(int k, int n_, std::vector<double>& matrix_) {
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
}

void konstantinov_i_gauss_jordan_method_mpi::NormalizeRow(int k, int n_, std::vector<double>& matrix_) {
  double diag = matrix_[(k * (n_ + 1)) + k];
  for (int j = k; j <= n_; ++j) {
    matrix_[(k * (n_ + 1)) + j] /= diag;
  }
}

void konstantinov_i_gauss_jordan_method_mpi::ProcessLocalMatrix(size_t local_size, int k, int n_,
                                                                std::vector<double>& localMatrix_,
                                                                const std::vector<double>& header_) {
  for (size_t i = 0; i < (local_size / (n_ + 1)); ++i) {
    double factor = localMatrix_[(i * (n_ + 1)) + k];
    for (int j = k; j <= n_; ++j) {
      localMatrix_[(i * (n_ + 1)) + j] -= header_[j] * factor;
    }
  }
}

void konstantinov_i_gauss_jordan_method_mpi::ProcessGaussStep(int k, int n_, std::vector<double>& matrix_,
                                                              std::vector<double>& header_,
                                                              std::vector<int>& sendCounts_,
                                                              std::vector<int>& displacements_,
                                                              boost::mpi::communicator& world_,
                                                              std::vector<double>& localMatrix_, bool is_forward) {
  if (world_.rank() == 0) {
    int offset = is_forward ? (n_ + 1) * (k + 1) : (n_ + 1) * k;
    size_t remainder_size = is_forward ? matrix_.size() - offset : offset;
    int elements_per_process = ((remainder_size / (n_ + 1)) / world_.size()) * (n_ + 1);
    int remainder = ((remainder_size / (n_ + 1)) % world_.size()) * (n_ + 1);

    sendCounts_ = std::vector<int>(world_.size(), elements_per_process);
    for (int i = 0; i < remainder / (n_ + 1); i++) {
      sendCounts_[i] += (n_ + 1);
    }

    displacements_ = std::vector<int>(world_.size(), is_forward ? offset : 0);
    for (int i = 1; i < world_.size(); ++i) {
      displacements_[i] = displacements_[i - 1] + sendCounts_[i - 1];
    }
  }

  boost::mpi::broadcast(world_, header_, 0);
  boost::mpi::broadcast(world_, sendCounts_, 0);
  boost::mpi::broadcast(world_, displacements_, 0);

  localMatrix_.resize(sendCounts_[world_.rank()]);
  boost::mpi::scatterv(world_, matrix_, sendCounts_, displacements_, localMatrix_.data(), sendCounts_[world_.rank()],
                       0);

  konstantinov_i_gauss_jordan_method_mpi::ProcessLocalMatrix(localMatrix_.size(), k, n_, localMatrix_, header_);

  boost::mpi::gatherv(world_, localMatrix_, matrix_.data(), sendCounts_, displacements_, 0);
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