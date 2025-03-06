#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/dudchenko_o_sum_values_by_cols/include/ops_sec.hpp"

TEST(dudchenko_o_sum_values_by_cols_seq, test_pipeline_run_seq) {
  int rows = 12000;
  int cols = 12000;

  std::vector<int> in(cols * rows, 0);
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      in[(i * cols) + j] = (i * cols) + j;
    }
  }
  std::vector<int> expect(cols, 0);
  for (int j = 0; j < cols; j++) {
    int tmp_sum = 0;
    for (int i = 0; i < rows; i++) {
      tmp_sum += in[(i * cols) + j];
    }
    expect[j] = tmp_sum;
  }
  std::vector<int> out(cols, 0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count = {static_cast<unsigned int>(in.size()), static_cast<unsigned int>(rows),
                                 static_cast<unsigned int>(cols)};
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(static_cast<unsigned int>(out.size()));

  auto test_task_seq = std::make_shared<dudchenko_o_sum_values_by_cols_seq::SumValByCols>(task_data_seq);

  ASSERT_TRUE(test_task_seq->ValidationImpl());
  test_task_seq->PreProcessingImpl();
  test_task_seq->RunImpl();
  test_task_seq->PostProcessingImpl();

  ASSERT_EQ(expect, out);
}

TEST(dudchenko_o_sum_values_by_cols_seq, test_task_run_seq) {
  int rows = 12000;
  int cols = 12000;

  std::vector<int> in(cols * rows, 0);
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      in[(i * cols) + j] = (i * cols) + j;
    }
  }
  std::vector<int> expect(cols, 0);
  for (int j = 0; j < cols; j++) {
    int tmp_sum = 0;
    for (int i = 0; i < rows; i++) {
      tmp_sum += in[(i * cols) + j];
    }
    expect[j] = tmp_sum;
  }
  std::vector<int> out(cols, 0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count = {static_cast<unsigned int>(in.size()), static_cast<unsigned int>(rows),
                                 static_cast<unsigned int>(cols)};
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(static_cast<unsigned int>(out.size()));

  auto test_task_seq = std::make_shared<dudchenko_o_sum_values_by_cols_seq::SumValByCols>(task_data_seq);

  ASSERT_TRUE(test_task_seq->ValidationImpl());
  test_task_seq->PreProcessingImpl();
  test_task_seq->RunImpl();
  test_task_seq->PostProcessingImpl();

  ASSERT_EQ(expect, out);
}
