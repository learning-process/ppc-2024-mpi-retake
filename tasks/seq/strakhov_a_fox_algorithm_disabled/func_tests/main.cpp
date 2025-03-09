#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/strakhov_a_fox_algorithm/include/ops_seq.hpp"

namespace {
std::vector<double> MultiplyMatrices(std::vector<double>& a, std::vector<double>& b, size_t n) {
  std::vector<double> c(a.size(), 0);
  for (unsigned int i = 0; i < n; ++i) {
    for (unsigned int j = 0; j < n; ++j) {
      for (unsigned int k = 0; k < n; ++k) {
        c[(i * n) + j] += (a[(i * n) + k] * b[(k * n) + j]);
      }
    }
  }
  return c;
}

std::vector<double> CreateRandomVal(double min_v, double max_v, size_t s) {
  std::random_device dev;
  std::mt19937 random(dev());
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dis(min_v, max_v);
  std::vector<double> ans(s, 0);
  for (size_t i = 0; i < s; i++) {
    ans[i] = dis(gen);
  }
  return ans;
}
}  // namespace

TEST(strakhov_a_fox_algorithm_seq, test_matmul_different_out_sizes) {
  constexpr size_t kCount = 2;

  // Create data
  std::vector<double> a = {1, 2, 3, 4};
  std::vector<double> b = {10, 11, 12, 15};
  std::vector<double> ans = {84, 90, 96, 201, 216, 231, 318, 342, 366};
  std::vector<double> out((kCount * kCount) + 7, 0);
  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(a.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(b.data()));
  task_data_seq->inputs_count.emplace_back(kCount);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  strakhov_a_fox_algorithm_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), false);
}

TEST(strakhov_a_fox_algorithm_seq, test_matmul_zero) {
  constexpr size_t kCount = 0;

  // Create data
  std::vector<double> a = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<double> b = {10, 11, 12, 13, 14, 15, 16, 17, 18};
  std::vector<double> ans = {84, 90, 96, 201, 216, 231, 318, 342, 366};
  std::vector<double> out(kCount * kCount, 0);
  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(a.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(b.data()));
  task_data_seq->inputs_count.emplace_back(kCount);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  strakhov_a_fox_algorithm_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), false);
}

TEST(strakhov_a_fox_algorithm_seq, test_matmul_one) {
  constexpr size_t kCount = 1;

  // Create data
  std::vector<double> a = {1};
  std::vector<double> b = {10};
  std::vector<double> ans = {10};
  std::vector<double> out(kCount * kCount, 0);
  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(a.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(b.data()));
  task_data_seq->inputs_count.emplace_back(kCount);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  strakhov_a_fox_algorithm_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(ans, out);
}

TEST(strakhov_a_fox_algorithm_seq, test_matmul_2x2) {
  constexpr size_t kCount = 2;

  // Create data
  std::vector<double> a = {1.1, 2.4, 3.7, 4.3};
  std::vector<double> b = {5.5, 6.1, 7.7, 8.4};
  std::vector<double> ans = {24.53, 26.87, 53.46, 58.69};
  std::vector<double> out(kCount * kCount, 0);
  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(a.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(b.data()));
  task_data_seq->inputs_count.emplace_back(kCount);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  strakhov_a_fox_algorithm_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(ans, out);
}

TEST(strakhov_a_fox_algorithm_seq, test_matmul_3x3) {
  constexpr size_t kCount = 3;

  // Create data
  std::vector<double> a = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<double> b = {10, 11, 12, 13, 14, 15, 16, 17, 18};
  std::vector<double> ans = {84, 90, 96, 201, 216, 231, 318, 342, 366};
  std::vector<double> out(kCount * kCount, 0);
  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(a.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(b.data()));
  task_data_seq->inputs_count.emplace_back(kCount);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  strakhov_a_fox_algorithm_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(ans, out);
}

TEST(strakhov_a_fox_algorithm_seq, test_matmul_4x4) {
  constexpr size_t kCount = 4;
  // Create data
  std::vector<double> a(kCount * kCount, 0);
  for (size_t i = 0; i < a.size(); i++) {
    a[i] = (double)(i + 1);
  }
  std::vector<double> b(kCount * kCount, 0);
  for (size_t i = 17; i < b.size() + 17; i++) {
    b[i - 17] = (double)(i);
  }
  std::vector<double> ans = {250, 260, 270, 280, 618, 644, 670, 696, 986, 1028, 1070, 1112, 1354, 1412, 1470, 1528};
  std::vector<double> out(kCount * kCount, 0);
  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(a.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(b.data()));
  task_data_seq->inputs_count.emplace_back(kCount);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());
  // Create Task
  strakhov_a_fox_algorithm_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(ans, out);
}

TEST(strakhov_a_fox_algorithm_seq, test_matmul_5x5) {
  constexpr size_t kCount = 5;
  // Create data
  std::vector<double> a(kCount * kCount, 0);
  for (size_t i = 0; i < a.size(); i++) {
    a[i] = (double)(i + 1);
  }
  std::vector<double> b(kCount * kCount, 0);
  for (size_t i = 0; i < b.size(); i++) {
    b[i] = (double)(i + 26);
  }
  std::vector<double> ans = {590,  605,  620,  635,  650,  1490, 1530, 1570, 1610, 1650, 2390, 2455, 2520,
                             2585, 2650, 3290, 3380, 3470, 3560, 3650, 4190, 4305, 4420, 4535, 4650};
  std::vector<double> out(kCount * kCount, 0);
  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(a.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(b.data()));
  task_data_seq->inputs_count.emplace_back(kCount);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());
  // Create Task
  strakhov_a_fox_algorithm_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(ans, out);
}

TEST(strakhov_a_fox_algorithm_seq, test_matmul_100x100_random) {
  constexpr size_t kCount = 100;
  // Create data
  std::vector<double> a = CreateRandomVal(-100, 100, kCount * kCount);
  std::vector<double> b = CreateRandomVal(-100, 100, kCount * kCount);
  std::vector<double> ans = MultiplyMatrices(a, b, kCount);
  std::vector<double> out(kCount * kCount, 0);
  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(a.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(b.data()));
  task_data_seq->inputs_count.emplace_back(kCount);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());
  // Create Task
  strakhov_a_fox_algorithm_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  for (size_t i = 0; i < out.size(); i++) {
    ASSERT_FLOAT_EQ(ans[i], out[i]);
  }
}
