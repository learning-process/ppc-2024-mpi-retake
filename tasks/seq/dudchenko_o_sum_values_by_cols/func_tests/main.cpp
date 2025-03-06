#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/dudchenko_o_sum_values_by_cols/include/ops_sec.hpp"

TEST(dudchenko_o_sum_values_by_cols_seq, test_sum_2x2_matrix) {
  const int rows = 2;
  const int cols = 2;

  std::vector<int> in = {1, 2, 3, 4};
  std::vector<int> expect = {4, 6};
  std::vector<int> out(cols, 0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count = {static_cast<unsigned int>(in.size()), static_cast<unsigned int>(rows),
                                 static_cast<unsigned int>(cols)};
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(static_cast<unsigned int>(out.size()));

  dudchenko_o_sum_values_by_cols_seq::SumValByCols sum_val_by_cols(task_data_seq);
  ASSERT_TRUE(sum_val_by_cols.ValidationImpl());
  sum_val_by_cols.PreProcessingImpl();
  sum_val_by_cols.RunImpl();
  sum_val_by_cols.PostProcessingImpl();
  ASSERT_EQ(expect, out);
}

TEST(dudchenko_o_sum_values_by_cols_seq, test_sum_3x3_matrix) {
  const int rows = 3;
  const int cols = 3;

  std::vector<int> in = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<int> expect = {12, 15, 18};
  std::vector<int> out(cols, 0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count = {static_cast<unsigned int>(in.size()), static_cast<unsigned int>(rows),
                                 static_cast<unsigned int>(cols)};
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(static_cast<unsigned int>(out.size()));

  dudchenko_o_sum_values_by_cols_seq::SumValByCols sum_val_by_cols(task_data_seq);
  ASSERT_TRUE(sum_val_by_cols.ValidationImpl());
  sum_val_by_cols.PreProcessingImpl();
  sum_val_by_cols.RunImpl();
  sum_val_by_cols.PostProcessingImpl();
  ASSERT_EQ(expect, out);
}
