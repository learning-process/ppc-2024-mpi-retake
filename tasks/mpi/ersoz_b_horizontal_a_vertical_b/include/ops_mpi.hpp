#ifndef ERSOZ_B_HORIZONTAL_A_VERTICAL_B_HPP
#define ERSOZ_B_HORIZONTAL_A_VERTICAL_B_HPP

#include <cstddef>
#include <vector>

std::vector<int> getRandomMatrix(std::size_t row_count, std::size_t column_count);

std::vector<int> getSequentialOperations(const std::vector<int>& matrix1, const std::vector<int>& matrix2,
                                         std::size_t A_rows, std::size_t A_cols, std::size_t B_cols);

std::vector<int> getParallelOperations(const std::vector<int>& matrix1, const std::vector<int>& matrix2,
                                       std::size_t A_rows, std::size_t A_cols);

#endif  // ERSOZ_B_HORIZONTAL_A_VERTICAL_B_HPP
