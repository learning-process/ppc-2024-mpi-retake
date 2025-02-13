#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/khokhlov_a_sum_values_by_rows/include/ops_sec.hpp"

namespace khokhlov_a_sum_values_by_rows_seq {
namespace {
std::vector<int> GetRandomMatrix(int size) {
  int sz = size;
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vec(sz);
  for (int i = 0; i < sz; i++) {
    vec[i] = (int)(gen() % 100);
  }
  return vec;
}
}  // namespace
}  // namespace khokhlov_a_sum_values_by_rows_seq

TEST(khokhlov_a_sum_values_by_rows_seq, Validation_test) {
  const int rows = 1;
  const int cols = 1;

  // create data
  std::vector<int> in(rows * cols, 0);
  std::vector<int> expect(rows, 0);
  std::vector<int> out(rows, 0);

  // create task data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->inputs_count.emplace_back(rows);
  task_data_seq->inputs_count.emplace_back(cols);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // crate task
  khokhlov_a_sum_values_by_rows_seq::SumValByRows sum_val_by_rows(task_data_seq);
  ASSERT_TRUE(sum_val_by_rows.ValidationImpl());
}

TEST(khokhlov_a_sum_values_by_rows_seq, test_sum_empty) {
  const int rows = 0;
  const int cols = 0;

  // create data
  std::vector<int> in = {};
  std::vector<int> expect;
  std::vector<int> out = {};

  // create task data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->inputs_count.emplace_back(rows);
  task_data_seq->inputs_count.emplace_back(cols);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // crate task
  khokhlov_a_sum_values_by_rows_seq::SumValByRows sum_val_by_rows(task_data_seq);
  ASSERT_TRUE(sum_val_by_rows.ValidationImpl());
  sum_val_by_rows.PreProcessingImpl();
  sum_val_by_rows.RunImpl();
  sum_val_by_rows.PostProcessingImpl();
  ASSERT_EQ(expect, out);
}

TEST(khokhlov_a_sum_values_by_rows_seq, test_sum_1x1_matrix) {
  const int rows = 1;
  const int cols = 1;

  // create data
  std::vector<int> in = {3};
  std::vector<int> expect = {3};
  std::vector<int> out(rows, 0);

  // create task data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->inputs_count.emplace_back(rows);
  task_data_seq->inputs_count.emplace_back(cols);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // crate task
  khokhlov_a_sum_values_by_rows_seq::SumValByRows sum_val_by_rows(task_data_seq);
  ASSERT_TRUE(sum_val_by_rows.ValidationImpl());
  sum_val_by_rows.PreProcessingImpl();
  sum_val_by_rows.RunImpl();
  sum_val_by_rows.PostProcessingImpl();
  ASSERT_EQ(expect, out);
}

TEST(khokhlov_a_sum_values_by_rows_seq, test_sum_1x_matrix) {
  const int rows = 1;
  const int cols = 5;

  // create data
  std::vector<int> in = {1, 2, 3, 4, 5};
  std::vector<int> expect = {15};
  std::vector<int> out(rows, 0);

  // create task data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->inputs_count.emplace_back(rows);
  task_data_seq->inputs_count.emplace_back(cols);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // crate task
  khokhlov_a_sum_values_by_rows_seq::SumValByRows sum_val_by_rows(task_data_seq);
  ASSERT_TRUE(sum_val_by_rows.ValidationImpl());
  sum_val_by_rows.PreProcessingImpl();
  sum_val_by_rows.RunImpl();
  sum_val_by_rows.PostProcessingImpl();
  ASSERT_EQ(expect, out);
}

TEST(khokhlov_a_sum_values_by_rows_seq, test_sum_x1_matrix) {
  const int rows = 5;
  const int cols = 1;

  // create data
  std::vector<int> in = {1, 2, 3, 4, 5};
  std::vector<int> expect = {1, 2, 3, 4, 5};
  std::vector<int> out(rows, 0);

  // create task data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->inputs_count.emplace_back(rows);
  task_data_seq->inputs_count.emplace_back(cols);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // crate task
  khokhlov_a_sum_values_by_rows_seq::SumValByRows sum_val_by_rows(task_data_seq);
  ASSERT_TRUE(sum_val_by_rows.ValidationImpl());
  sum_val_by_rows.PreProcessingImpl();
  sum_val_by_rows.RunImpl();
  sum_val_by_rows.PostProcessingImpl();
  ASSERT_EQ(expect, out);
}

TEST(khokhlov_a_sum_values_by_rows_seq, test_sum_2x2_matrix) {
  const int rows = 2;
  const int cols = 2;

  // create data
  std::vector<int> in = {1, 2, 3, 4};
  std::vector<int> expect = {3, 7};
  std::vector<int> out(rows, 0);

  // create task data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->inputs_count.emplace_back(rows);
  task_data_seq->inputs_count.emplace_back(cols);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // crate task
  khokhlov_a_sum_values_by_rows_seq::SumValByRows sum_val_by_rows(task_data_seq);
  ASSERT_TRUE(sum_val_by_rows.ValidationImpl());
  sum_val_by_rows.PreProcessingImpl();
  sum_val_by_rows.RunImpl();
  sum_val_by_rows.PostProcessingImpl();
  ASSERT_EQ(expect, out);
}

TEST(khokhlov_a_sum_values_by_rows_seq, test_sum_2x4_matrix) {
  const int rows = 2;
  const int cols = 4;

  // create data
  std::vector<int> in = {1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<int> expect = {10, 26};
  std::vector<int> out(rows, 0);

  // create task data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->inputs_count.emplace_back(rows);
  task_data_seq->inputs_count.emplace_back(cols);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // crate task
  khokhlov_a_sum_values_by_rows_seq::SumValByRows sum_val_by_rows(task_data_seq);
  ASSERT_TRUE(sum_val_by_rows.ValidationImpl());
  sum_val_by_rows.PreProcessingImpl();
  sum_val_by_rows.RunImpl();
  sum_val_by_rows.PostProcessingImpl();
  ASSERT_EQ(expect, out);
}

TEST(khokhlov_a_sum_values_by_rows_seq, test_sum_4x2_matrix) {
  const int rows = 4;
  const int cols = 2;

  // create data
  std::vector<int> in = {1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<int> expect = {3, 7, 11, 15};
  std::vector<int> out(rows, 0);

  // create task data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->inputs_count.emplace_back(rows);
  task_data_seq->inputs_count.emplace_back(cols);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // crate task
  khokhlov_a_sum_values_by_rows_seq::SumValByRows sum_val_by_rows(task_data_seq);
  ASSERT_TRUE(sum_val_by_rows.ValidationImpl());
  sum_val_by_rows.PreProcessingImpl();
  sum_val_by_rows.RunImpl();
  sum_val_by_rows.PostProcessingImpl();
  ASSERT_EQ(expect, out);
}

TEST(khokhlov_a_sum_values_by_rows_seq, test_sum_4x3_matrix_with_negative_elements) {
  const int rows = 4;
  const int cols = 3;

  // create data
  std::vector<int> in = {1, 2, -3, 3, 4, -6, 5, 6, -9, 7, 8, -12};
  std::vector<int> expect = {0, 1, 2, 3};
  std::vector<int> out(rows, 0);

  // create task data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->inputs_count.emplace_back(rows);
  task_data_seq->inputs_count.emplace_back(cols);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // crate task
  khokhlov_a_sum_values_by_rows_seq::SumValByRows sum_val_by_rows(task_data_seq);
  ASSERT_TRUE(sum_val_by_rows.ValidationImpl());
  sum_val_by_rows.PreProcessingImpl();
  sum_val_by_rows.RunImpl();
  sum_val_by_rows.PostProcessingImpl();
  ASSERT_EQ(expect, out);
}

TEST(khokhlov_a_sum_values_by_rows_seq, test_sum_100x100_matrix) {
  const int rows = 100;
  const int cols = 100;

  // create data
  std::vector<int> in(cols * rows, 2);
  std::vector<int> expect(rows, cols * 2);
  std::vector<int> out(rows, 0);

  // create task data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->inputs_count.emplace_back(rows);
  task_data_seq->inputs_count.emplace_back(cols);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // crate task
  khokhlov_a_sum_values_by_rows_seq::SumValByRows sum_val_by_rows(task_data_seq);
  ASSERT_TRUE(sum_val_by_rows.ValidationImpl());
  sum_val_by_rows.PreProcessingImpl();
  sum_val_by_rows.RunImpl();
  sum_val_by_rows.PostProcessingImpl();
  ASSERT_EQ(expect, out);
}

TEST(khokhlov_a_sum_values_by_rows_seq, test_sum_1rand_00x100_matrix) {
  const int rows = 100;
  const int cols = 100;

  // create data
  std::vector<int> in = khokhlov_a_sum_values_by_rows_seq::GetRandomMatrix(rows * cols);
  std::vector<int> expect(rows, 0);
  for (int i = 0; i < rows; i++) {
    int tmp_sum = 0;
    for (int j = 0; j < cols; j++) {
      tmp_sum += in[(i * cols) + j];
    }
    expect[i] += tmp_sum;
  }
  std::vector<int> out(rows, 0);

  // create task data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->inputs_count.emplace_back(rows);
  task_data_seq->inputs_count.emplace_back(cols);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // crate task
  khokhlov_a_sum_values_by_rows_seq::SumValByRows sum_val_by_rows(task_data_seq);
  ASSERT_TRUE(sum_val_by_rows.ValidationImpl());
  sum_val_by_rows.PreProcessingImpl();
  sum_val_by_rows.RunImpl();
  sum_val_by_rows.PostProcessingImpl();
  ASSERT_EQ(expect, out);
}