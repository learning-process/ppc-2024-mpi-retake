
#include "mpi/vasenkov_a_gauss_jordan/include/ops_mpi.hpp"

#include <algorithm>
#include <cmath>
#include <vector>


#define EPSILON 1e-9

namespace vasenkov_a_gauss_jordan_mpi {

bool GaussJordanMethodParallelMPI::ValidationImpl() {
  if (world.rank() != 0) return true;

  int n_val = *reinterpret_cast<int *>(task_data->inputs[1]);
  int matrix_size = task_data->inputs_count[0];
  auto *matrix_data = reinterpret_cast<double *>(task_data->inputs[0]);

  if (n_val * (n_val + 1) != matrix_size) return false;

  std::vector<double> tempMatrix(n_val * n_val);
  for (int i = 0; i < n_val; ++i) {
    for (int j = 0; j < n_val; ++j) {
      tempMatrix[i * n_val + j] = matrix_data[i * (n_val + 1) + j];
    }
  }

  for (int k = 0; k < n_val; ++k) {
    double max = fabs(tempMatrix[k * n_val + k]);
    int maxRow = k;
    for (int i = k + 1; i < n_val; ++i) {
      if (fabs(tempMatrix[i * n_val + k]) > max) {
        max = fabs(tempMatrix[i * n_val + k]);
        maxRow = i;
      }
    }
    if (fabs(tempMatrix[maxRow * n_val + k]) < EPSILON) return false;

    if (maxRow != k) {
      for (int j = 0; j < n_val; ++j) {
        std::swap(tempMatrix[k * n_val + j], tempMatrix[maxRow * n_val + j]);
      }
    }

    for (int i = k + 1; i < n_val; ++i) {
      double factor = tempMatrix[i * n_val + k] / tempMatrix[k * n_val + k];
      for (int j = k; j < n_val; ++j) {
        tempMatrix[i * n_val + j] -= factor * tempMatrix[k * n_val + j];
      }
    }
  }
  return true;
}

bool GaussJordanMethodParallelMPI::PreProcessingImpl() {
  if (world.rank() == 0) {
    auto *matrix_data = reinterpret_cast<double *>(task_data->inputs[0]);
    int matrix_size = task_data->inputs_count[0];
    n = *reinterpret_cast<int *>(task_data->inputs[1]);
    matrix.assign(matrix_data, matrix_data + matrix_size);
  }
  boost::mpi::broadcast(world, n, 0);
  return true;
}

bool GaussJordanMethodParallelMPI::RunImpl() {
  for (int k = 0; k < n; ++k) {
    if (world.rank() == 0) {
      if (fabs(matrix[k * (n + 1) + k]) < EPSILON) {
        int swap_row = -1;
        for (int i = k + 1; i < n; ++i) {
          if (fabs(matrix[i * (n + 1) + k]) > EPSILON) {
            swap_row = i;
            break;
          }
        }
        if (swap_row == -1) {
          solve = false;
          break;
        }
        for (int col = 0; col <= n; ++col) {
          std::swap(matrix[k * (n + 1) + col], matrix[swap_row * (n + 1) + col]);
        }
      }

      double pivot = matrix[k * (n + 1) + k];
      for (int j = k; j <= n; ++j) {
        matrix[k * (n + 1) + j] /= pivot;
      }
    }

    boost::mpi::broadcast(world, solve, 0);
    if (!solve) return false;

    if (world.rank() == 0) {
      for (int i = 0; i < n; ++i) {
        if (i != k) {
          double factor = matrix[i * (n + 1) + k];
          for (int j = k; j <= n; ++j) {
            matrix[i * (n + 1) + j] -= factor * matrix[k * (n + 1) + j];
          }
          matrix[i * (n + 1) + k] = 0.0;
        }
      }
    }
  }

  return true;
}

bool GaussJordanMethodParallelMPI::PostProcessingImpl() {
  if (!solve) return false;

  if (world.rank() == 0) {
    auto *output_data = reinterpret_cast<double *>(task_data->outputs[0]);
    std::copy(matrix.begin(), matrix.end(), output_data);
  }
  return true;
}

bool GaussJordanMethodSequentialMPI::ValidationImpl() {
  int n_val = *reinterpret_cast<int *>(task_data->inputs[1]);
  int matrix_size = task_data->inputs_count[0];
  return n_val * (n_val + 1) == matrix_size;
}

bool GaussJordanMethodSequentialMPI::PreProcessingImpl() {
  auto *matrix_data = reinterpret_cast<double *>(task_data->inputs[0]);
  int matrix_size = task_data->inputs_count[0];
  n_size = *reinterpret_cast<int *>(task_data->inputs[1]);
  sys_matrix.assign(matrix_data, matrix_data + matrix_size);
  return true;
}

bool GaussJordanMethodSequentialMPI::RunImpl() {
  for (int k = 0; k < n_size; ++k) {
    if (sys_matrix[k * (n_size + 1) + k] == 0.0) {
      int swap_row = -1;
      for (int i = k + 1; i < n_size; ++i) {
        if (std::abs(sys_matrix[i * (n_size + 1) + k]) > 1e-6) {
          swap_row = i;
          break;
        }
      }
      if (swap_row == -1) return false;

      for (int col = 0; col <= n_size; ++col) {
        std::swap(sys_matrix[k * (n_size + 1) + col], sys_matrix[swap_row * (n_size + 1) + col]);
      }
    }

    const double pivot = sys_matrix[k * (n_size + 1) + k];
    for (int j = k; j <= n_size; ++j) {
      sys_matrix[k * (n_size + 1) + j] /= pivot;
    }

    for (int i = 0; i < n_size; ++i) {
      if (i != k && sys_matrix[i * (n_size + 1) + k] != 0.0) {
        const double factor = sys_matrix[i * (n_size + 1) + k];
        for (int j = k; j <= n_size; ++j) {
          sys_matrix[i * (n_size + 1) + j] -= factor * sys_matrix[k * (n_size + 1) + j];
        }
        sys_matrix[i * (n_size + 1) + k] = 0.0;
      }
    }
  }

  return true;
}

bool GaussJordanMethodSequentialMPI::PostProcessingImpl() {
  auto *output_data = reinterpret_cast<double *>(task_data->outputs[0]);
  std::copy(sys_matrix.begin(), sys_matrix.end(), output_data);
  return true;
}

}  // namespace vasenkov_a_gauss_jordan_mpi