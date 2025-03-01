#include "mpi/Konstantinov_I_Gauss_Jordan_method/include/ops_mpi.hpp"
#include <algorithm>
#include <boost/serialization/array.hpp>
#include <boost/serialization/vector.hpp>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <numeric>
#include <vector>

#define EPSILON 1e-9

namespace konstantinov_i_gauss_jordan_method_mpi {

  bool IsNonSingularSystem(const std::vector<double>& A, int n) {
    std::vector<double> tempMatrix(n * n);

    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        tempMatrix[i * n + j] = A[i * (n + 1) + j];
      }
    }

    for (int k = 0; k < n; ++k) {
      double max = fabs(tempMatrix[k * n + k]);
      int maxRow = k;
      for (int i = k + 1; i < n; ++i) {
        if (fabs(tempMatrix[i * n + k]) > max) {
          max = fabs(tempMatrix[i * n + k]);
          maxRow = i;
        }
      }

      if (fabs(tempMatrix[maxRow * n + k]) < EPSILON) {
        return false;
      }

      if (maxRow != k) {
        for (int j = 0; j < n; ++j) {
          std::swap(tempMatrix[k * n + j], tempMatrix[maxRow * n + j]);
        }
      }

      for (int i = k + 1; i < n; ++i) {
        double factor = tempMatrix[i * n + k] / tempMatrix[k * n + k];
        for (int j = k; j < n; ++j) {
          tempMatrix[i * n + j] -= factor * tempMatrix[k * n + j];
        }
      }
    }
    return true;
  }

}  // namespace konstantinov_i_gauss_jordan_method_mpi

std::vector<double> konstantinov_i_gauss_jordan_method_mpi::ProcessMatrix(int n, int k,
                                                                       const std::vector<double>& matrix) {
  std::vector<double> result_vec(n * (n - k + 1));

  for (int i = 0; i < (n - k + 1); i++) {
    result_vec[i] = matrix[(n + 1) * k + k + i];
  }

  for (int i = 0; i < k; i++) {
    for (int j = 0; j < (n - k + 1); j++) {
      result_vec[(n - k + 1) * (i + 1) + j] = matrix[i * (n + 1) + k + j];
    }
  }

  for (int i = k + 1; i < n; i++) {
    for (int j = 0; j < (n - k + 1); j++) {
      result_vec[(n - k + 1) * i + j] = matrix[i * (n + 1) + k + j];
    }
  }

  return result_vec;
}

void konstantinov_i_gauss_jordan_method_mpi::CalcSizesDispls(int n, int k, int world_size, std::vector<int>& sizes,
                                                          std::vector<int>& displs) {
  int r = n - 1;
  int c = n - k;
  sizes.resize(world_size, 0);
  displs.resize(world_size, 0);

  if (world_size > r) {
    for (int i = 0; i < r; ++i) {
      sizes[i] = c;
      displs[i] = i * c;
    }
  } else {
    int a = r / world_size;
    int b = r % world_size;

    int offset = 0;
    for (int i = 0; i < world_size; ++i) {
      if (b-- > 0) {
        sizes[i] = (a + 1) * c;
      } else {
        sizes[i] = a * c;
      }
      displs[i] = offset;
      offset += sizes[i];
    }
  }
}

std::vector<std::pair<int, int>> konstantinov_i_gauss_jordan_method_mpi::GetIndicies(int rows, int cols) {
  std::vector<std::pair<int, int>> indicies;
  indicies.reserve(rows * cols);

  for (int i = 1; i < rows; ++i) {
    for (int j = 1; j < cols; ++j) {
      indicies.emplace_back(i, j);
    }
  }
  return indicies;
}

void konstantinov_i_gauss_jordan_method_mpi::UpdateMatrix(int n, int k, std::vector<double>& matrix,
                                                       const std::vector<double>& iter_result) {
  for (int i = 0; i < k; i++) {
    for (int j = 0; j < (n - k); j++) {
      matrix[i * (n + 1) + k + 1 + j] = iter_result[i * (n - k) + j];
    }
  }

  for (int i = k + 1; i < n; i++) {
    for (int j = 0; j < (n - k); j++) {
      matrix[i * (n + 1) + k + 1 + j] = iter_result[(i - 1) * (n - k) + j];
    }
  }

  for (int i = k + 1; i < n + 1; i++) matrix[k * (n + 1) + i] /= matrix[k * (n + 1) + k];

  for (int i = 0; i < n; i++) {
    matrix[i * (n + 1) + k] = 0;
  }

  matrix[k * (n + 1) + k] = 1;
}

bool konstantinov_i_gauss_jordan_method_mpi::GaussJordanMethodMPI::ValidationImpl() {
  if (world.rank() != 0) {
    return true;
  }
  int n_val = *reinterpret_cast<int*>(task_data->inputs[1]);
  int matrix_size = task_data->inputs_count[0];
  auto* matrix_data = reinterpret_cast<double*>(task_data->inputs[0]);

  if (n_val * (n_val + 1) == matrix_size) {
    std::vector<double> temp_matrix(matrix_size);
    temp_matrix.assign(matrix_data, matrix_data + matrix_size);
    return konstantinov_i_gauss_jordan_method_mpi::IsNonSingularSystem(temp_matrix, n_val);
  }
  return false;
}

bool konstantinov_i_gauss_jordan_method_mpi::GaussJordanMethodMPI::PreProcessingImpl() {

  if (world.rank() == 0) {
    auto* matrix_data = reinterpret_cast<double*>(task_data->inputs[0]);
    int matrix_size = task_data->inputs_count[0];

    n = *reinterpret_cast<int*>(task_data->inputs[1]);

    matrix.assign(matrix_data, matrix_data + matrix_size);
  }

  return true;
}

bool konstantinov_i_gauss_jordan_method_mpi::GaussJordanMethodMPI::RunImpl() {

  boost::mpi::broadcast(world, n, 0);

  for (int k = 0; k < n; k++) {
    if (world.rank() == 0) {
      if (matrix[k * (n + 1) + k] == 0) {
        int change;
        for (change = k + 1; change < n; change++) {
          if (matrix[change * (n + 1) + k] != 0) {
            for (int col = 0; col < (n + 1); col++) {
              std::swap(matrix[k * (n + 1) + col], matrix[change * (n + 1) + col]);
            }
            break;
          }
        }
        if (change == n) {
          solve = false;
        }
      }

      if (solve) {
        iter_matrix = konstantinov_i_gauss_jordan_method_mpi::ProcessMatrix(n, k, matrix);

        konstantinov_i_gauss_jordan_method_mpi::CalcSizesDispls(n, k, world.size(), sizes, displs);
        indicies = konstantinov_i_gauss_jordan_method_mpi::GetIndicies(n, n - k + 1);

        iter_result.resize((n - 1) * (n - k));
      }
    }
    boost::mpi::broadcast(world, solve, 0);
    if (!solve) return false;
    boost::mpi::broadcast(world, sizes, 0);
    boost::mpi::broadcast(world, iter_matrix, 0);

    int local_size = sizes[world.rank()];
    std::vector<std::pair<int, int>> local_indicies(local_size);
    if (world.rank() == 0) {
      boost::mpi::scatterv(world, indicies.data(), sizes, displs, local_indicies.data(), local_size, 0);
    } else {
      boost::mpi::scatterv(world, local_indicies.data(), local_size, 0);
    }

    std::vector<double> local_result;
    local_result.reserve(local_size);
    for (int ind = 0; ind < local_size; ind++) {
      auto [i, j] = local_indicies[ind];
      double rel = iter_matrix[0];
      double nel = iter_matrix[i * (n - k + 1) + j];
      double a = iter_matrix[j];
      double b = iter_matrix[i * (n - k + 1)];
      double res = nel - (a * b) / rel;
      local_result[ind] = res;
    }

    if (world.rank() == 0) {
      boost::mpi::gatherv(world, local_result.data(), local_size, iter_result.data(), sizes, displs, 0);
    } else {
      boost::mpi::gatherv(world, local_result.data(), local_size, 0);
    }

    if (world.rank() == 0) {
      konstantinov_i_gauss_jordan_method_mpi::UpdateMatrix(n, k, matrix, iter_result);
    }
  }

  return true;
}

bool konstantinov_i_gauss_jordan_method_mpi::GaussJordanMethodMPI::PostProcessingImpl() {
  if (!solve) {
    return false;
  }
  if (world.rank() == 0) {
    auto* output_data = reinterpret_cast<double*>(task_data->outputs[0]);
    std::copy(matrix.begin(), matrix.end(), output_data);
  }

  return true;
}

bool konstantinov_i_gauss_jordan_method_mpi::GaussJordanMethodSeq::ValidationImpl() {
  int n_val = *reinterpret_cast<int*>(task_data->inputs[1]);
  int matrix_size = task_data->inputs_count[0];
  return n_val * (n_val + 1) == matrix_size;
}

bool konstantinov_i_gauss_jordan_method_mpi::GaussJordanMethodSeq::PreProcessingImpl() {
  auto* matrix_data = reinterpret_cast<double*>(task_data->inputs[0]);
  int matrix_size = task_data->inputs_count[0];
  n = *reinterpret_cast<int*>(task_data->inputs[1]);
  matrix.assign(matrix_data, matrix_data + matrix_size);

  return true;
}

bool konstantinov_i_gauss_jordan_method_mpi::GaussJordanMethodSeq::RunImpl() {
  for (int k = 0; k < n; k++) {
    if (matrix[k * (n + 1) + k] == 0) {
      int change;
      for (change = k + 1; change < n; change++) {
        if (matrix[change * (n + 1) + k] != 0) {
          for (int col = 0; col < (n + 1); col++) {
            std::swap(matrix[k * (n + 1) + col], matrix[change * (n + 1) + col]);
          }
          break;
        }
      }
      if (change == n) {
        solve = false;
        return false;
      }
    }

    std::vector<double> iter_matrix = konstantinov_i_gauss_jordan_method_mpi::ProcessMatrix(n, k, matrix);

    std::vector<double> iter_result((n - 1) * (n - k));

    int ind = 0;
    for (int i = 1; i < n; ++i) {
      for (int j = 1; j < n - k + 1; ++j) {
        double rel = iter_matrix[0];
        double nel = iter_matrix[i * (n - k + 1) + j];
        double a = iter_matrix[j];
        double b = iter_matrix[i * (n - k + 1)];
        double res = nel - (a * b) / rel;
        iter_result[ind++] = res;
      }
    }

    konstantinov_i_gauss_jordan_method_mpi::UpdateMatrix(n, k, matrix, iter_result);
  }

  return true;
}

bool konstantinov_i_gauss_jordan_method_mpi::GaussJordanMethodSeq::PostProcessingImpl() {
  if (!solve) {
    return false;
  }
  auto* output_data = reinterpret_cast<double*>(task_data->outputs[0]);
  std::copy(matrix.begin(), matrix.end(), output_data);

  return true;
}