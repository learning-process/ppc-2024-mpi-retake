#include <gtest/gtest.h>

#include <memory>
#include <numeric>
#include <vector>

#include "seq/shishkarev_a_gaussian_method_horizontal_strip_pattern/include/ops_seq.hpp"

TEST(shishkarev_a_gaussian_method_horizontal_strip_pattern_seq, test_for_empty_matrix) {
  const int cols = 0;
  const int rows = 0;

  std::vector<double> matrix;
  std::vector<double> res;

  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
  taskDataSeq->inputs_count = {static_cast<unsigned int>(matrix.size()), cols, rows};
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(res.data()));
  taskDataSeq->outputs_count.emplace_back(res.size());

  auto task =
      std::make_shared<shishkarev_a_gaussian_method_horizontal_strip_pattern_seq::MPIGaussHorizontalSequential<double>>(
          taskDataSeq);
  ASSERT_FALSE(task->ValidationImpl());
}

TEST(shishkarev_a_gaussian_method_horizontal_strip_pattern_seq, test_for_matrix_with_one_element) {
  const int cols = 1;
  const int rows = 1;

  std::vector<double> matrix = {1};
  std::vector<double> res;

  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
  taskDataSeq->inputs_count = {static_cast<unsigned int>(matrix.size()), cols, rows};
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(res.data()));
  taskDataSeq->outputs_count.emplace_back(res.size());

  auto task =
      std::make_shared<shishkarev_a_gaussian_method_horizontal_strip_pattern_seq::MPIGaussHorizontalSequential<double>>(
          taskDataSeq);
  ASSERT_FALSE(task->ValidationImpl());
}

TEST(shishkarev_a_gaussian_method_horizontal_strip_pattern_seq, test_not_square_matrix) {
  const int cols = 5;
  const int rows = 2;

  std::vector<double> matrix = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  std::vector<double> res(cols - 1, 0);

  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
  taskDataSeq->inputs_count = {static_cast<unsigned int>(matrix.size()), cols, rows};
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(res.data()));
  taskDataSeq->outputs_count.emplace_back(res.size());

  auto task =
      std::make_shared<shishkarev_a_gaussian_method_horizontal_strip_pattern_seq::MPIGaussHorizontalSequential<double>>(
          taskDataSeq);
}
