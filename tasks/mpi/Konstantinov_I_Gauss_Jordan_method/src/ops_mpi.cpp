#include "mpi/Konstantinov_I_Gauss_Jordan_method/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/collectives/broadcast.hpp>
#include <boost/mpi/collectives/reduce.hpp>
#include <boost/mpi/collectives/scatterv.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/serialization/vector.hpp>
#include <cmath>
#include <ranges>
#include <utility>
#include <vector>

#define EPSILON 1e-9

bool konstantinov_i_gauss_jordan_method_mpi::IsNonSingularSystem(const std::vector<double>& a, int n) {
  std::vector<double> temp_matrix(n * n);

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      temp_matrix[(i * n) + j] = a[(i * (n + 1)) + j];
    }
  }

  for (int k = 0; k < n; ++k) {
    double max = fabs(temp_matrix[(k * n) + k]);
    int max_row = k;
    for (int i = k + 1; i < n; ++i) {
      if (fabs(temp_matrix[(i * n) + k]) > max) {
        max = fabs(temp_matrix[(i * n) + k]);
        max_row = i;
      }
    }

    if (fabs(temp_matrix[(max_row * n) + k]) < EPSILON) {
      return false;
    }

    if (max_row != k) {
      for (int j = 0; j < n; ++j) {
        std::swap(temp_matrix[(k * n) + j], temp_matrix[(max_row * n) + j]);
      }
    }

    for (int i = k + 1; i < n; ++i) {
      double factor = temp_matrix[(i * n) + k] / temp_matrix[(k * n) + k];
      for (int j = k; j < n; ++j) {
        temp_matrix[(i * n) + j] -= factor * temp_matrix[(k * n) + j];
      }
    }
  }
  return true;
}

std::vector<double> konstantinov_i_gauss_jordan_method_mpi::ProcessMatrix(int n, int k,
                                                                          const std::vector<double>& matrix) {
  std::vector<double> result_vec(n * (n - k + 1));

  for (int i = 0; i < (n - k + 1); i++) {
    result_vec[i] = matrix[((n + 1) * k) + k + i];
  }

  for (int i = 0; i < k; i++) {
    for (int j = 0; j < (n - k + 1); j++) {
      result_vec[((n - k + 1) * (i + 1)) + j] = matrix[(i * (n + 1)) + k + j];
    }
  }

  for (int i = k + 1; i < n; i++) {
    for (int j = 0; j < (n - k + 1); j++) {
      result_vec[((n - k + 1) * i) + j] = matrix[(i * (n + 1)) + k + j];
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
      matrix[(i * (n + 1)) + k + 1 + j] = iter_result[(i * (n - k)) + j];
    }
  }

  for (int i = k + 1; i < n; i++) {
    for (int j = 0; j < (n - k); j++) {
      matrix[(i * (n + 1)) + k + 1 + j] = iter_result[((i - 1) * (n - k)) + j];
    }
  }

  for (int i = k + 1; i < n + 1; i++) {
    matrix[(k * (n + 1)) + i] /= matrix[(k * (n + 1)) + k];
  }

  for (int i = 0; i < n; i++) {
    matrix[(i * (n + 1)) + k] = 0;
  }

  matrix[(k * (n + 1)) + k] = 1;
}

bool konstantinov_i_gauss_jordan_method_mpi::SwapRowsIfZero(int n, int k, std::vector<double>& matrix, bool& solve) {
  if (matrix[(k * (n + 1)) + k] != 0) return true;
  for (int change = k + 1; change < n; change++) {
    if (matrix[(change * (n + 1)) + k] != 0) {
      SwapRows(n, k, change, matrix);
      return true;
    }
  }
  solve = false;
  return false;
}

void konstantinov_i_gauss_jordan_method_mpi::SwapRows(int n, int k, int change, std::vector<double>& matrix) {
  for (int col = 0; col < (n + 1); col++) {
    std::swap(matrix[(k * (n + 1)) + col], matrix[(change * (n + 1)) + col]);
  }
}

void konstantinov_i_gauss_jordan_method_mpi::PrepareIteration(int n, int k, const std::vector<double>& matrix,
                                                              std::vector<double>& iter_matrix, std::vector<int>& sizes,
                                                              std::vector<int>& displs,
                                                              std::vector<std::pair<int, int>>& indicies,
                                                              std::vector<double>& iter_result,
                                                              boost::mpi::communicator& world) {
  iter_matrix = konstantinov_i_gauss_jordan_method_mpi::ProcessMatrix(n, k, matrix);
  konstantinov_i_gauss_jordan_method_mpi::CalcSizesDispls(n, k, world.size(), sizes, displs);
  indicies = konstantinov_i_gauss_jordan_method_mpi::GetIndicies(n, n - k + 1);
  iter_result.resize((n - 1) * (n - k));
}

bool konstantinov_i_gauss_jordan_method_mpi::BroadcastSolve(boost::mpi::communicator& world, bool solve) {
  boost::mpi::broadcast(world, solve, 0);
  return solve;
}

void konstantinov_i_gauss_jordan_method_mpi::BroadcastData(boost::mpi::communicator& world, std::vector<int>& sizes,
                                                           std::vector<double>& iter_matrix) {
  boost::mpi::broadcast(world, sizes, 0);
  boost::mpi::broadcast(world, iter_matrix, 0);
}

std::vector<std::pair<int, int>> konstantinov_i_gauss_jordan_method_mpi::ScatterIndices(
    boost::mpi::communicator& world, std::vector<std::pair<int, int>>& indicies, std::vector<int>& sizes,
    std::vector<int>& displs, int local_size) {
  std::vector<std::pair<int, int>> local_indicies(local_size);
  if (world.rank() == 0) {
    boost::mpi::scatterv(world, indicies.data(), sizes, displs, local_indicies.data(), local_size, 0);
  } else {
    boost::mpi::scatterv(world, local_indicies.data(), local_size, 0);
  }
  return local_indicies;
}

std::vector<double> konstantinov_i_gauss_jordan_method_mpi::CalculateLocalResult(
    int local_size, std::vector<std::pair<int, int>>& local_indicies, const std::vector<double>& iter_matrix, int n,
    int k) {
  std::vector<double> local_result(local_size);
  double rel = iter_matrix[0];
  for (int ind = 0; ind < local_size; ind++) {
    auto [i, j] = local_indicies[ind];
    double nel = iter_matrix[(i * (n - k + 1)) + j];
    double a = iter_matrix[j];
    double b = iter_matrix[i * (n - k + 1)];
    local_result[ind] = nel - ((a * b) / rel);
  }
  return local_result;
}

void konstantinov_i_gauss_jordan_method_mpi::GatherResults(boost::mpi::communicator& world,
                                                           std::vector<double>& local_result,
                                                           std::vector<double>& iter_result, std::vector<int>& sizes,
                                                           std::vector<int>& displs) {
  if (world.rank() == 0) {
    boost::mpi::gatherv(world, local_result.data(), local_result.size(), iter_result.data(), sizes, displs, 0);
  } else {
    boost::mpi::gatherv(world, local_result.data(), local_result.size(), 0);
  }
}

bool konstantinov_i_gauss_jordan_method_mpi::GaussJordanMethodMPI::ValidationImpl() {
  if (world.rank() != 0) {
    return true;
  }
  int n_val = *reinterpret_cast<int*>(task_data->inputs[1]);
  int matrix_size = static_cast<int>(task_data->inputs_count[0]);
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
    int matrix_size = static_cast<int>(task_data->inputs_count[0]);

    n = *reinterpret_cast<int*>(task_data->inputs[1]);

    matrix.assign(matrix_data, matrix_data + matrix_size);
  }

  return true;
}

bool konstantinov_i_gauss_jordan_method_mpi::GaussJordanMethodMPI::RunImpl() {
  boost::mpi::broadcast(world, n, 0);

  for (int k = 0; k < n; k++) {
    if (world.rank() == 0) {
      if (!SwapRowsIfZero(n, k, matrix, solve)) {
        return false;
      }
      if (solve) {
        PrepareIteration(n, k, matrix, iter_matrix, sizes, displs, indicies, iter_result, world);
      }
    }

    if (!BroadcastSolve(world, solve)) {
      return false;
    }
    BroadcastData(world, sizes, iter_matrix);

    int local_size = sizes[world.rank()];
    auto local_indicies = ScatterIndices(world, indicies, sizes, displs, local_size);

    auto local_result = CalculateLocalResult(local_size, local_indicies, iter_matrix, n, k);
    GatherResults(world, local_result, iter_result, sizes, displs);

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
    std::ranges::copy(matrix, output_data);
  }

  return true;
}

bool konstantinov_i_gauss_jordan_method_mpi::GaussJordanMethodSeq::ValidationImpl() {
  int n_val = *reinterpret_cast<int*>(task_data->inputs[1]);
  int matrix_size = static_cast<int>(task_data->inputs_count[0]);
  return n_val * (n_val + 1) == matrix_size;
}

bool konstantinov_i_gauss_jordan_method_mpi::GaussJordanMethodSeq::PreProcessingImpl() {
  auto* matrix_data = reinterpret_cast<double*>(task_data->inputs[0]);
  int matrix_size = static_cast<int>(task_data->inputs_count[0]);
  n = *reinterpret_cast<int*>(task_data->inputs[1]);
  matrix.assign(matrix_data, matrix_data + matrix_size);

  return true;
}

bool konstantinov_i_gauss_jordan_method_mpi::GaussJordanMethodSeq::RunImpl() {
  for (int k = 0; k < n; k++) {
    if (matrix[(k * (n + 1)) + k] == 0) {
      int change = 0;
      for (change = k + 1; change < n; change++) {
        if (matrix[(change * (n + 1)) + k] != 0) {
          for (int col = 0; col < (n + 1); col++) {
            std::swap(matrix[(k * (n + 1)) + col], matrix[(change * (n + 1)) + col]);
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
        double nel = iter_matrix[(i * (n - k + 1)) + j];
        double a = iter_matrix[j];
        double b = iter_matrix[i * (n - k + 1)];
        double res = nel - ((a * b) / rel);
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
  std::ranges::copy(matrix, output_data);

  return true;
}