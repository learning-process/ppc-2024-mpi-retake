#include <mpi.h>

#include <random>
#include <vector>

#include "mpi/ersoz_b_rectangular_method_integration/include/ops_mpi.hpp"

std::vector<int> getRandomMatrix(std::size_t row_count, std::size_t column_count) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::vector<int> matrix(row_count * column_count);
  for (std::size_t i = 0; i < row_count; ++i) {
    for (std::size_t j = 0; j < column_count; ++j) {
      matrix[i * column_count + j] = gen() % 100;
    }
  }
  return matrix;
}

std::vector<int> getSequentialOperations(const std::vector<int>& matrix1, const std::vector<int>& matrix2,
                                         std::size_t A_rows, std::size_t A_cols, std::size_t B_cols) {
  std::vector<int> result(A_rows * B_cols, 0);
  for (std::size_t i = 0; i < A_rows; ++i) {
    for (std::size_t j = 0; j < B_cols; ++j) {
      int sum = 0;
      for (std::size_t k = 0; k < A_cols; ++k) {
        sum += matrix1[i * A_cols + k] * matrix2[k * B_cols + j];
      }
      result[i * B_cols + j] = sum;
    }
  }
  return result;
}

std::vector<int> getParallelOperations(const std::vector<int>& matrix1, const std::vector<int>& matrix2,
                                       std::size_t A_rows, std::size_t A_cols) {
  int size, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (size > static_cast<int>(A_rows) || size == 1) {
    return rank == 0 ? getSequentialOperations(matrix1, matrix2, A_rows, A_cols, A_rows) : std::vector<int>{};
  }

  std::size_t B_cols = A_rows;
  std::size_t local_count = A_rows / size;
  std::size_t remaining = A_rows % size;

  if (rank == 0) {
    for (int proc = 1; proc < size; proc++) {
      std::size_t offset_A = proc * local_count * A_cols;
      std::size_t offset_B = proc * local_count;
      if (remaining != 0) {
        offset_A += remaining * A_cols;
        offset_B += remaining;
      }
      std::vector<int> subMatrixB(local_count * A_cols, 0);
      for (std::size_t i = 0; i < local_count; i++) {
        for (std::size_t j = 0; j < B_cols; j++) {
          subMatrixB[i + j * local_count] = matrix2[offset_B + i + j * A_cols];
        }
      }
      MPI_Send(matrix1.data() + offset_A, static_cast<int>(local_count * A_cols), MPI_INT, proc, 1, MPI_COMM_WORLD);
      MPI_Send(subMatrixB.data(), static_cast<int>(subMatrixB.size()), MPI_INT, proc, 2, MPI_COMM_WORLD);
    }
  }

  std::size_t local_A_rows = (rank == 0) ? (local_count + remaining) : local_count;
  std::size_t local_A_size = local_A_rows * A_cols;
  std::vector<int> local_matrixA(local_A_size, 0);
  std::vector<int> local_matrixB(local_A_rows * A_cols + 1, 0);

  if (rank == 0) {
    for (std::size_t i = 0; i < local_A_size; i++) {
      local_matrixA[i] = matrix1[i];
    }
    for (std::size_t i = 0; i < local_A_rows; i++) {
      for (std::size_t j = 0; j < B_cols; j++) {
        local_matrixB[i + j * local_A_rows] = matrix2[i + j * local_A_rows];
      }
    }
  } else {
    MPI_Status status;
    MPI_Recv(local_matrixA.data(), static_cast<int>(local_A_size), MPI_INT, 0, 1, MPI_COMM_WORLD, &status);
    MPI_Recv(local_matrixB.data(), static_cast<int>(local_A_rows * A_cols), MPI_INT, 0, 2, MPI_COMM_WORLD, &status);
  }
  local_matrixB[local_A_rows * A_cols] = (rank == 0) ? 0 : rank * local_count + remaining;

  std::vector<int> local_partial_result =
      getSequentialOperations(local_matrixA, local_matrixB, local_A_rows, A_cols, B_cols);
  std::vector<int> global_result(A_rows * B_cols, 0);
  MPI_Reduce(local_partial_result.data(), global_result.data(), static_cast<int>(global_result.size()), MPI_INT,
             MPI_SUM, 0, MPI_COMM_WORLD);

  return global_result;
}
