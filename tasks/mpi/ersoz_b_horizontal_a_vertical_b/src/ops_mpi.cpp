#define OMPI_SKIP_MPICXX

#include "mpi/ersoz_b_horizontal_a_vertical_b/include/ops_mpi.hpp"

#include <mpi.h>

#include <cstddef>
#include <random>
#include <stdexcept>
#include <vector>

std::vector<int> GetRandomMatrix(std::size_t row_count, std::size_t column_count) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::vector<int> matrix(row_count * column_count);
  for (std::size_t i = 0; i < row_count; ++i) {
    for (std::size_t j = 0; j < column_count; ++j) {
      matrix[(i * column_count) + j] = static_cast<int>(gen() % 100);
    }
  }
  return matrix;
}

std::vector<int> GetSequentialOperations(const std::vector<int>& matrix1, const std::vector<int>& matrix2,
                                         std::size_t a_rows, std::size_t a_cols, std::size_t b_cols) {
  if (matrix1.size() != a_rows * a_cols) {
    throw std::invalid_argument("Invalid dimensions for matrix1");
  }
  if (matrix2.size() != a_cols * b_cols) {
    throw std::invalid_argument("Invalid dimensions for matrix2");
  }
  std::vector<int> result(a_rows * b_cols, 0);
  for (std::size_t i = 0; i < a_rows; ++i) {
    for (std::size_t j = 0; j < b_cols; ++j) {
      int sum = 0;
      for (std::size_t k = 0; k < a_cols; ++k) {
        sum += matrix1[(i * a_cols) + k] * matrix2[(k * b_cols) + j];
      }
      result[(i * b_cols) + j] = sum;
    }
  }
  return result;
}

std::vector<int> GetParallelOperations(const std::vector<int>& matrix1, const std::vector<int>& matrix2,
                                       std::size_t a_rows, std::size_t a_cols) {
  const std::size_t b_cols = a_rows;
  if (matrix1.size() != a_rows * a_cols) {
    throw std::invalid_argument("Invalid dimensions for matrix1");
  }
  if (matrix2.size() != a_cols * b_cols) {
    throw std::invalid_argument("Invalid dimensions for matrix2");
  }

  int size = 0;
  int rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (size == 1) {
    return GetSequentialOperations(matrix1, matrix2, a_rows, a_cols, b_cols);
  }

  std::size_t rows_per_proc = a_rows / size;
  std::size_t remainder = a_rows % size;
  std::size_t local_rows = rows_per_proc + (static_cast<std::size_t>(rank) < remainder ? 1 : 0);

  std::vector<int> sendcounts(size);
  std::vector<int> displs(size);
  for (int i = 0; i < size; i++) {
    std::size_t rows_i = rows_per_proc + (static_cast<std::size_t>(i) < remainder ? 1 : 0);
    sendcounts[i] = static_cast<int>(rows_i * a_cols);
    if (i == 0) {
      displs[i] = 0;
    } else {
      displs[i] = displs[i - 1] + sendcounts[i - 1];
    }
  }

  std::vector<int> local_matrix1(local_rows * a_cols);
  MPI_Scatterv(matrix1.data(), sendcounts.data(), displs.data(), MPI_INT, local_matrix1.data(), sendcounts[rank],
               MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<int> local_matrix2(matrix2);
  MPI_Bcast(local_matrix2.data(), static_cast<int>(a_cols * b_cols), MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<int> local_result(local_rows * b_cols, 0);
  for (std::size_t i = 0; i < local_rows; i++) {
    for (std::size_t j = 0; j < b_cols; j++) {
      int sum = 0;
      for (std::size_t k = 0; k < a_cols; k++) {
        sum += local_matrix1[(i * a_cols) + k] * local_matrix2[(k * b_cols) + j];
      }
      local_result[(i * b_cols) + j] = sum;
    }
  }

  std::vector<int> recvcounts(size);
  std::vector<int> rdispls(size);
  for (int i = 0; i < size; i++) {
    std::size_t rows_i = rows_per_proc + (static_cast<std::size_t>(i) < remainder ? 1 : 0);
    recvcounts[i] = static_cast<int>(rows_i * b_cols);
    if (i == 0) {
      rdispls[i] = 0;
    } else {
      rdispls[i] = rdispls[i - 1] + recvcounts[i - 1];
    }
  }

  std::vector<int> global_result;
  if (rank == 0) {
    global_result.resize(a_rows * b_cols);
  }
  MPI_Gatherv(local_result.data(), static_cast<int>(local_result.size()), MPI_INT, global_result.data(),
              recvcounts.data(), rdispls.data(), MPI_INT, 0, MPI_COMM_WORLD);
  return global_result;
}
