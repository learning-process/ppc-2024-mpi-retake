#include "mpi/vasenkov_a_gauss_jordan/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <cmath>
#include <vector>

#define EPSILON 1e-9

namespace vasenkov_a_gauss_jordan_mpi {

bool GaussJordanMethodParallelMPI::ValidationImpl() {
  if (world_.rank() != 0) {
    return true;
  }

  int n_val = *reinterpret_cast<int *>(task_data->inputs[1]);
  int matrix_size = static_cast<int>(task_data->inputs_count[0]);
  auto *matrix_data = reinterpret_cast<double *>(task_data->inputs[0]);

  if (n_val * (n_val + 1) != matrix_size) {
    return false;
  }

  std::vector<double> temp_matrix(n_val * n_val);
  for (int i = 0; i < n_val; ++i) {
    for (int j = 0; j < n_val; ++j) {
      temp_matrix[(i * n_val) + j] = matrix_data[(i * (n_val + 1)) + j];
    }
  }

  for (int k = 0; k < n_val; ++k) {
    double max = fabs(temp_matrix[(k * n_val) + k]);
    int max_row = k;
    for (int i = k + 1; i < n_val; ++i) {
      if (fabs(temp_matrix[(i * n_val) + k]) > max) {
        max = fabs(temp_matrix[(i * n_val) + k]);
        max_row = i;
      }
    }
    if (fabs(temp_matrix[(max_row * n_val) + k]) < EPSILON) {
      return false;
    }

    if (max_row != k) {
      for (int j = 0; j < n_val; ++j) {
        std::swap(temp_matrix[(k * n_val) + j], temp_matrix[(max_row * n_val) + j]);
      }
    }

    for (int i = k + 1; i < n_val; ++i) {
      double factor = temp_matrix[(i * n_val) + k] / temp_matrix[(k * n_val) + k];
      for (int j = k; j < n_val; ++j) {
        temp_matrix[(i * n_val) + j] -= factor * temp_matrix[(k * n_val) + j];
      }
    }
  }
  return true;
}

bool GaussJordanMethodParallelMPI::PreProcessingImpl() {
  if (world_.rank() == 0) {
    auto *matrix_data = reinterpret_cast<double *>(task_data->inputs[0]);
    int matrix_size = static_cast<int>(task_data->inputs_count[0]);
    n_size_ = *reinterpret_cast<int *>(task_data->inputs[1]);
    sys_matrix_.assign(matrix_data, matrix_data + matrix_size);
  }
  boost::mpi::broadcast(world_, n_size_, 0);
  return true;
}

bool GaussJordanMethodParallelMPI::RunImpl() {
  for (int k = 0; k < n_size_; ++k) {
    if (world_.rank() == 0) {
      if (fabs(sys_matrix_[(k * (n_size_ + 1)) + k]) < EPSILON) {
        int swap_row = -1;
        for (int i = k + 1; i < n_size_; ++i) {
          if (fabs(sys_matrix_[(i * (n_size_ + 1)) + k]) > EPSILON) {
            swap_row = i;
            break;
          }
        }
        if (swap_row == -1) {
          solve_ = false;
          break;
        }
        for (int col = 0; col <= n_size_; ++col) {
          std::swap(sys_matrix_[(k * (n_size_ + 1)) + col], sys_matrix_[(swap_row * (n_size_ + 1)) + col]);
        }
      }

      double pivot = sys_matrix_[(k * (n_size_ + 1)) + k];
      for (int j = k; j <= n_size_; ++j) {
        sys_matrix_[(k * (n_size_ + 1)) + j] /= pivot;
      }
    }

    boost::mpi::broadcast(world_, solve_, 0);
    if (!solve_) {
      return false;
    }

    if (world_.rank() == 0) {
      for (int i = 0; i < n_size_; ++i) {
        if (i != k) {
          double factor = sys_matrix_[(i * (n_size_ + 1)) + k];
          for (int j = k; j <= n_size_; ++j) {
            sys_matrix_[(i * (n_size_ + 1)) + j] -= factor * sys_matrix_[(k * (n_size_ + 1)) + j];
          }
          sys_matrix_[(i * (n_size_ + 1)) + k] = 0.0;
        }
      }
    }
  }

  return true;
}

bool GaussJordanMethodParallelMPI::PostProcessingImpl() {
  if (!solve_) {
    return false;
  }

  if (world_.rank() == 0) {
    auto *output_data = reinterpret_cast<double *>(task_data->outputs[0]);
    std::ranges::copy(sys_matrix_, output_data);
  }
  return true;
}

bool GaussJordanMethodSequentialMPI::ValidationImpl() {
  int n_val = *reinterpret_cast<int *>(task_data->inputs[1]);
  int matrix_size = static_cast<int>(task_data->inputs_count[0]);
  return n_val * (n_val + 1) == matrix_size;
}

bool GaussJordanMethodSequentialMPI::PreProcessingImpl() {
  auto *matrix_data = reinterpret_cast<double *>(task_data->inputs[0]);
  int matrix_size = static_cast<int>(task_data->inputs_count[0]);
  n_size_ = *reinterpret_cast<int *>(task_data->inputs[1]);
  sys_matrix_.assign(matrix_data, matrix_data + matrix_size);
  return true;
}

bool GaussJordanMethodSequentialMPI::RunImpl() {
  for (int k = 0; k < n_size_; ++k) {
    if (!EnsureNonZeroPivot(k)) {
      return false;
    }
    NormalizeRow(k);
    EliminateColumn(k);
  }
  return true;
}

bool GaussJordanMethodSequentialMPI::EnsureNonZeroPivot(int k) {
  if (sys_matrix_[(k * (n_size_ + 1)) + k] == 0.0) {
    int swap_row = FindSwapRow(k);
    if (swap_row == -1) {
      return false;
    }
    SwapRows(k, swap_row);
  }
  return true;
}

int GaussJordanMethodSequentialMPI::FindSwapRow(int k) {
  for (int i = k + 1; i < n_size_; ++i) {
    if (std::abs(sys_matrix_[(i * (n_size_ + 1)) + k]) > 1e-6) {
      return i;
    }
  }
  return -1;
}

void GaussJordanMethodSequentialMPI::SwapRows(int row1, int row2) {
  for (int col = 0; col <= n_size_; ++col) {
    std::swap(sys_matrix_[(row1 * (n_size_ + 1)) + col], sys_matrix_[(row2 * (n_size_ + 1)) + col]);
  }
}

void GaussJordanMethodSequentialMPI::NormalizeRow(int k) {
  const double pivot = sys_matrix_[(k * (n_size_ + 1)) + k];
  for (int j = k; j <= n_size_; ++j) {
    sys_matrix_[(k * (n_size_ + 1)) + j] /= pivot;
  }
}

void GaussJordanMethodSequentialMPI::EliminateColumn(int k) {
  for (int i = 0; i < n_size_; ++i) {
    if (i != k && sys_matrix_[(i * (n_size_ + 1)) + k] != 0.0) {
      const double factor = sys_matrix_[(i * (n_size_ + 1)) + k];
      for (int j = k; j <= n_size_; ++j) {
        sys_matrix_[(i * (n_size_ + 1)) + j] -= factor * sys_matrix_[(k * (n_size_ + 1)) + j];
      }
      sys_matrix_[(i * (n_size_ + 1)) + k] = 0.0;
    }
  }
}

bool GaussJordanMethodSequentialMPI::PostProcessingImpl() {
  auto *output_data = reinterpret_cast<double *>(task_data->outputs[0]);
  std::ranges::copy(sys_matrix_, output_data);
  return true;
}

}  // namespace vasenkov_a_gauss_jordan_mpi