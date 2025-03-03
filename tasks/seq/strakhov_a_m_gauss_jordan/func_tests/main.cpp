#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/strakhov_a_m_gauss_jordan/include/ops_seq.hpp"

namespace {
std::vector<double> GenRandomVector(size_t size, int min, int max) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> dis(min, max);
  std::vector<double> random_vector(size);
  for (size_t i = 0; i < size; i++) {
    random_vector[i] = (double)(dis(gen));
  }

  return random_vector;
}
}  // namespace

TEST(strakhov_a_m_gauss_jordan_seq, test_mat_val_zero_size) {
  constexpr size_t kCount = 0;

  // Create data
  std::vector<double> in = {};
  std::vector<double> out(kCount, 0);
  std::vector<double> ans = {};

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(kCount + 1);
  task_data_seq->inputs_count.emplace_back(kCount);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  strakhov_a_m_gauss_jordan_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), false);
}

TEST(strakhov_a_m_gauss_jordan_seq, test_mat_val_zero_col) {
  constexpr size_t kCount = 2;

  // Create data
  std::vector<double> in = {0, 1, 0, 0, 1, 0};
  std::vector<double> out(kCount, 0);
  std::vector<double> ans = {0, 0};

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(kCount + 1);
  task_data_seq->inputs_count.emplace_back(kCount);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  strakhov_a_m_gauss_jordan_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), false);
}

TEST(strakhov_a_m_gauss_jordan_seq, test_mat_val_zero_row) {
  constexpr size_t kCount = 2;

  // Create data
  std::vector<double> in = {0, 0, 0, 1, 1, 1};
  std::vector<double> out(kCount, 0);
  std::vector<double> ans = {0, 0};

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(kCount + 1);
  task_data_seq->inputs_count.emplace_back(kCount);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  strakhov_a_m_gauss_jordan_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), false);
}

TEST(strakhov_a_m_gauss_jordan_seq, test_mat_1) {
  constexpr size_t kCount = 1;

  // Create data
  std::vector<double> in = {1, 2};
  std::vector<double> out(kCount, 0);
  std::vector<double> ans = {2};

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(kCount + 1);
  task_data_seq->inputs_count.emplace_back(kCount);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  strakhov_a_m_gauss_jordan_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(ans, out);
}

TEST(strakhov_a_m_gauss_jordan_seq, test_mat_2) {
  constexpr size_t kCount = 2;

  // Create data
  std::vector<double> in = {4, -1, 7, 1, 1, 8};
  std::vector<double> out(kCount, 0);
  std::vector<double> ans = {3, 5};

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(kCount + 1);
  task_data_seq->inputs_count.emplace_back(kCount);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  strakhov_a_m_gauss_jordan_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(ans, out);
}

TEST(strakhov_a_m_gauss_jordan_seq, test_mat_3) {
  constexpr size_t kCount = 3;

  // Create data
  std::vector<double> in = {1, 2, 3, 5, 0, 1, 2, 4, 0, 2, 7, 11};
  std::vector<double> out(kCount, 0);
  std::vector<double> ans = {-2, 2, 1};

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(kCount + 1);
  task_data_seq->inputs_count.emplace_back(kCount);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  strakhov_a_m_gauss_jordan_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(ans, out);
}

TEST(strakhov_a_m_gauss_jordan_seq, test_mat_3_negt_ans) {
  constexpr size_t kCount = 3;

  // Create data
  std::vector<double> in = {-2, 1, 0, -3, 3, -1, 0, 2, 2, 0, 1, 2};
  std::vector<double> out(kCount, 0);
  std::vector<double> ans = {-1, -5, 4};

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(kCount + 1);
  task_data_seq->inputs_count.emplace_back(kCount);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  strakhov_a_m_gauss_jordan_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(ans, out);
}

TEST(strakhov_a_m_gauss_jordan_seq, test_mat_4) {
  constexpr size_t kCount = 4;

  // Create data
  std::vector<double> in = {1, -2, 1, 4, 16, 4, 3, -2, 1, 8, 2, -2, 2, -2, -4, 4, -8, 3, 2, 5};
  std::vector<double> out(kCount, 0);
  std::vector<double> ans = {1, 2, 3, 4};

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(kCount + 1);
  task_data_seq->inputs_count.emplace_back(kCount);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  strakhov_a_m_gauss_jordan_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  for (size_t i = 0; i < kCount; i++) {
    ASSERT_FLOAT_EQ(ans[i], out[i]);
  }
}

TEST(strakhov_a_m_gauss_jordan_seq, test_mat_5) {
  constexpr size_t kCount = 5;

  // Create data
  std::vector<double> in = {3,  2, 1, 0, 0, 10, 1,  1,  -1, 2, 2, 18, -5, -1, -1,
                            -1, 3, 1, 5, 3, 1,  -1, -1, 5,  1, 2, 4,  -3, 1,  10};
  std::vector<double> out(kCount, 0);
  std::vector<double> ans = {1, 2, 3, 4, 5};

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(kCount + 1);
  task_data_seq->inputs_count.emplace_back(kCount);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  strakhov_a_m_gauss_jordan_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  for (size_t i = 0; i < kCount; i++) {
    ASSERT_FLOAT_EQ(ans[i], out[i]);
  }
}

TEST(strakhov_a_m_gauss_jordan_seq, test_mat_7_random) {
  constexpr size_t kCount = 7;

  // Create data
  std::vector<double> in = GenRandomVector(kCount * (kCount + 1), -5, 55);
  std::vector<double> ans = {1, 2, 3, 4, 5, 6, 7};
  for (size_t i = 0; i < kCount; i++) {
    double sum = 0;
    for (size_t j = 0; j < kCount; j++) {
      sum += ans[j] * in[((kCount + 1) * i) + j];
    }
    in[((kCount + 1) * (i + 1)) - 1] = sum;
  }
  std::vector<double> out(kCount, 0);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(kCount + 1);
  task_data_seq->inputs_count.emplace_back(kCount);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  strakhov_a_m_gauss_jordan_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  for (size_t i = 0; i < kCount; i++) {
    ASSERT_FLOAT_EQ(ans[i], out[i]);
  }
}
