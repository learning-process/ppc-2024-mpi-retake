// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/opolin_d_simple_iteration_method/include/ops_seq.hpp"

namespace opolin_d_simple_iteration_method_seq {
namespace {
void GenerateTestData(size_t size, std::vector<double> &x, std::vector<double> &a, std::vector<double> &b) {
  std::srand(static_cast<unsigned>(std::time(nullptr)));

  x.resize(size);
  for (size_t i = 0; i < size; ++i) {
    x[i] = -10.0 + static_cast<double>(std::rand() % 1000) / 50.0;
  }

  a.resize(size * size, 0.0);
  for (size_t i = 0; i < size; ++i) {
    double sum = 0.0;
    for (size_t j = 0; j < size; ++j) {
      if (i != j) {
        a[(i * size) + j] = -1.0 + static_cast<double>(std::rand() % 1000) / 500.0;
        sum += std::abs(a[(i * size) + j]);
      }
    }
    a[(i * size) + i] = sum + 1.0;
  }
  b.resize(size, 0.0);
  for (size_t i = 0; i < size; ++i) {
    for (size_t j = 0; j < size; ++j) {
      b[i] += a[(i * size) + j] * x[j];
    }
  }
}
}  // namespace
}  // namespace opolin_d_simple_iteration_method_seq

TEST(opolin_d_simple_iteration_method_seq, test_small_system) {
  int size = 3;
  double epsilon = 1e-9;
  int max_iters = 1000;
  std::vector<double> expected;
  std::vector<double> a;
  std::vector<double> b;
  opolin_d_simple_iteration_method_seq::GenerateTestData(size, expected, a, b);

  std::vector<double> out(size, 0.0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
  task_data_seq->inputs_count.emplace_back(out.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&max_iters));
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  opolin_d_simple_iteration_method_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  for (int i = 0; i < size; ++i) {
    ASSERT_NEAR(expected[i], out[i], 1e-3);
  }
}

TEST(opolin_d_simple_iteration_method_seq, test_big_system) {
  int size = 100;
  double epsilon = 1e-9;
  int max_iters = 1000;
  std::vector<double> expected;
  std::vector<double> a;
  std::vector<double> b;
  opolin_d_simple_iteration_method_seq::GenerateTestData(size, expected, a, b);

  std::vector<double> out(size, 0.0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
  task_data_seq->inputs_count.emplace_back(out.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&max_iters));
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  opolin_d_simple_iteration_method_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  for (int i = 0; i < size; ++i) {
    ASSERT_NEAR(expected[i], out[i], 1e-3);
  }
}

TEST(opolin_d_simple_iteration_method_seq, test_negative_values) {
  int size = 3;
  double epsilon = 1e-9;
  int max_iters = 1000;
  std::vector<double> expected;
  std::vector<double> a;
  std::vector<double> b;

  a = {5.0, -1.0, 2.0, -1.0, 6.0, -1.0, 2.0, -1.0, 7.0};
  b = {-9.0, -8.0, -21.0};
  expected = {-1.0, -2.0, -3.0};

  std::vector<double> out(size, 0.0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
  task_data_seq->inputs_count.emplace_back(out.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&max_iters));
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  opolin_d_simple_iteration_method_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  for (int i = 0; i < size; ++i) {
    ASSERT_NEAR(expected[i], out[i], 1e-3);
  }
}

TEST(opolin_d_simple_iteration_method_seq, test_single_element) {
  int size = 1;
  double epsilon = 1e-9;
  int max_iters = 1000;
  std::vector<double> a = {4.0};
  std::vector<double> b = {8.0};
  std::vector<double> expected = {2.0};

  std::vector<double> out(size, 0.0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
  task_data_seq->inputs_count.emplace_back(out.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&max_iters));
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  opolin_d_simple_iteration_method_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  for (int i = 0; i < size; ++i) {
    ASSERT_NEAR(expected[i], out[i], 1e-3);
  }
}

TEST(opolin_d_simple_iteration_method_seq, test_no_dominance_matrix) {
  int size = 3;
  double epsilon = 1e-9;
  int max_iters = 1000;
  std::vector<double> a = {3.0, 2.0, 4.0, 1.0, 2.0, 4.0, 1.0, 2.0, 3.0};
  std::vector<double> b = {3.0, 2.0, 2.0};
  std::vector<double> out(size, 0.0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
  task_data_seq->inputs_count.emplace_back(out.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&max_iters));
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  opolin_d_simple_iteration_method_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.ValidationImpl(), false);
}

TEST(opolin_d_simple_iteration_method_seq, test_singular_matrix) {
  int size = 3;
  double epsilon = 1e-9;
  int max_iters = 1000;
  std::vector<double> a = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 5.0, 7.0, 9.0};
  std::vector<double> b = {1.0, 2.0, 3.0};

  std::vector<double> out(size, 0.0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
  task_data_seq->inputs_count.emplace_back(out.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&max_iters));
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  opolin_d_simple_iteration_method_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), false);
}

TEST(opolin_d_simple_iteration_method_seq, test_random_generated_data) {
  int size = 5;
  double epsilon = 1e-9;
  int max_iters = 1000;
  std::vector<double> expected;
  std::vector<double> a;
  std::vector<double> b;
  opolin_d_simple_iteration_method_seq::GenerateTestData(size, expected, a, b);

  std::vector<double> out(size, 0.0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
  task_data_seq->inputs_count.emplace_back(out.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&max_iters));
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  opolin_d_simple_iteration_method_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  for (int i = 0; i < size; ++i) {
    ASSERT_NEAR(expected[i], out[i], 1e-3);
  }
}

TEST(opolin_d_simple_iteration_method_seq, test_correct_input) {
  int size = 3;
  double epsilon = 1e-9;
  int max_iters = 1000;
  std::vector<double> expected;
  std::vector<double> a;
  std::vector<double> b;
  a = {4.0, 1.0, 2.0, 1.0, 5.0, 1.0, 2.0, 1.0, 5.0};
  b = {7.0, 7.0, 8.0};
  expected = {1.0, 1.0, 1.0};

  std::vector<double> out(size, 0.0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
  task_data_seq->inputs_count.emplace_back(out.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&max_iters));
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  opolin_d_simple_iteration_method_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  for (int i = 0; i < size; ++i) {
    ASSERT_NEAR(expected[i], out[i], 1e-3);
  }
}

TEST(opolin_d_simple_iteration_method_seq, test_simple_matrix) {
  int size = 3;
  double epsilon = 1e-9;
  int max_iters = 1000;
  std::vector<double> expected;
  std::vector<double> a;
  std::vector<double> b;
  a = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
  b = {1.0, 1.0, 1.0};
  expected = {1.0, 1.0, 1.0};

  std::vector<double> out(size, 0.0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
  task_data_seq->inputs_count.emplace_back(out.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&max_iters));
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  opolin_d_simple_iteration_method_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  for (int i = 0; i < size; ++i) {
    ASSERT_NEAR(expected[i], out[i], 1e-3);
  }
}