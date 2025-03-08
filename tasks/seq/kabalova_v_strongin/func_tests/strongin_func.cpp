#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>
#include <numbers>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/kabalova_v_strongin/include/strongin.h"

// Все минимумы найдены +- по графику.

TEST(kabalova_v_strongin_seq, x_square) {
  double left = 0;
  double right = 10;
  double res1 = 0;
  double res2 = 0;
  std::function<double(double)> f = [](double x) { return x * x; };
  double answer1 = 0;
  double answer2 = f(answer1);
  double eps = 0.0001;
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&left));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&right));
  task_data_seq->inputs_count.emplace_back(2);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&res1));
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&res2));
  task_data_seq->outputs_count.emplace_back(2);

  // Create Task
  kabalova_v_strongin_seq::TestTaskSequential test_task_sequential(task_data_seq, f);
  ASSERT_EQ(test_task_sequential.ValidationImpl(), true);
  test_task_sequential.PreProcessingImpl();
  test_task_sequential.RunImpl();
  test_task_sequential.PostProcessingImpl();
  EXPECT_NEAR(answer1, res1, eps);
  EXPECT_NEAR(answer2, res2, eps);
}

TEST(kabalova_v_strongin_seq, sin) {
  double left = -std::numbers::pi;
  double right = std::numbers::pi;
  double res1 = 0;
  double res2 = 0;
  std::function<double(double)> f = [](double x) { return std::sin(x); };
  double eps = 0.1;
  double answer1 = -std::numbers::pi / 2;
  double answer2 = f(answer1);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&left));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&right));
  task_data_seq->inputs_count.emplace_back(2);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&res1));
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&res2));
  task_data_seq->outputs_count.emplace_back(2);

  // Create Task
  kabalova_v_strongin_seq::TestTaskSequential test_task_sequential(task_data_seq, f);
  ASSERT_EQ(test_task_sequential.ValidationImpl(), true);
  test_task_sequential.PreProcessingImpl();
  test_task_sequential.RunImpl();
  test_task_sequential.PostProcessingImpl();
  EXPECT_NEAR(answer1, res1, eps);
  EXPECT_NEAR(answer2, res2, eps);
}

TEST(kabalova_v_strongin_seq, cos) {
  double left = -std::numbers::pi;
  double right = std::numbers::pi / 2;
  double res1 = 0;
  double res2 = 0;
  std::function<double(double)> f = [](double x) { return std::cos(x); };
  double eps = 0.1;
  double answer1 = -std::numbers::pi;
  double answer2 = f(answer1);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&left));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&right));
  task_data_seq->inputs_count.emplace_back(2);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&res1));
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&res2));
  task_data_seq->outputs_count.emplace_back(2);

  // Create Task
  kabalova_v_strongin_seq::TestTaskSequential test_task_sequential(task_data_seq, f);
  ASSERT_EQ(test_task_sequential.ValidationImpl(), true);
  test_task_sequential.PreProcessingImpl();
  test_task_sequential.RunImpl();
  test_task_sequential.PostProcessingImpl();
  EXPECT_NEAR(answer1, res1, eps);
  EXPECT_NEAR(answer2, res2, eps);
}

TEST(kabalova_v_strongin_seq, x_polynome) {
  double left = -10;
  double right = 0;
  double res1 = 0;
  double res2 = 0;
  std::function<double(double)> f = [](double x) { return (x * x * x * (-0.2465)) + (x * x * (-0.3147)) + 1.0; };
  double eps = 0.1;
  double answer1 = -0.8;
  double answer2 = f(answer1);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&left));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&right));
  task_data_seq->inputs_count.emplace_back(2);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&res1));
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&res2));
  task_data_seq->outputs_count.emplace_back(2);

  // Create Task
  kabalova_v_strongin_seq::TestTaskSequential test_task_sequential(task_data_seq, f);
  ASSERT_EQ(test_task_sequential.ValidationImpl(), true);
  test_task_sequential.PreProcessingImpl();
  test_task_sequential.RunImpl();
  test_task_sequential.PostProcessingImpl();
  EXPECT_NEAR(answer1, res1, eps);
  EXPECT_NEAR(answer2, res2, eps);
}
