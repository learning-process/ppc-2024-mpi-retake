#ifndef ERSOZ_B_HORIZONTAL_A_VERTICAL_B_MPI_HPP
#define ERSOZ_B_HORIZONTAL_A_VERTICAL_B_MPI_HPP

#include <cstddef>
#include <vector>

std::vector<int> GetRandomMatrix(std::size_t row_count, std::size_t column_count);

std::vector<int> GetSequentialOperations(const std::vector<int>& matrix1, const std::vector<int>& matrix2,
                                         std::size_t a_rows, std::size_t a_cols, std::size_t b_cols);

std::vector<int> GetParallelOperations(const std::vector<int>& matrix1, const std::vector<int>& matrix2,
                                       std::size_t a_rows, std::size_t a_cols);

#endif  // ERSOZ_B_HORIZONTAL_A_VERTICAL_B_MPI_HPP
