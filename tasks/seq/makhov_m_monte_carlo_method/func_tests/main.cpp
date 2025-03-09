#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <numbers>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/makhov_m_monte_carlo_method/include/ops_seq.hpp"

TEST(makhov_m_monte_carlo_method_seq, func_is_x_pow2) {
  // Create data
  std::function<double(const std::vector<double> &)> f = [](const std::vector<double> &x) { return x[0] * x[0]; };
  int num_samples = 1000000;
  std::vector<std::pair<double, double>> limits = {{0.0, 1.0}};
  double *answer_ptr = nullptr;
  double reference = 0.33;

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.push_back(reinterpret_cast<uint8_t *>(&f));
  task_data_seq->inputs.push_back(reinterpret_cast<uint8_t *>(&num_samples));
  task_data_seq->inputs.push_back(reinterpret_cast<uint8_t *>(limits.data()));
  task_data_seq->inputs_count.emplace_back(1);
  task_data_seq->inputs_count.emplace_back(1);
  task_data_seq->inputs_count.emplace_back(1);  // Integral dimension info
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(answer_ptr));
  task_data_seq->outputs_count.emplace_back(1);

  // Create Task
  makhov_m_monte_carlo_method_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  uint8_t *answer_data = task_data_seq->outputs[0];
  double retrieved_value = NAN;
  std::memcpy(&retrieved_value, answer_data, sizeof(double));
  double truncated_value = std::round(retrieved_value * 100) / 100;
  EXPECT_EQ(reference, truncated_value);
}

TEST(makhov_m_monte_carlo_method_seq, func_is_sinx) {
  // Create data

  std::function<double(const std::vector<double> &)> f = [](const std::vector<double> &x) { return std::sin(x[0]); };
  int num_samples = 1000000;
  std::vector<std::pair<double, double>> limits = {{0.0, std::numbers::pi}};
  double *answer_ptr = nullptr;
  double reference = 2.0;

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.push_back(reinterpret_cast<uint8_t *>(&f));
  task_data_seq->inputs.push_back(reinterpret_cast<uint8_t *>(&num_samples));
  task_data_seq->inputs.push_back(reinterpret_cast<uint8_t *>(limits.data()));
  task_data_seq->inputs_count.emplace_back(1);
  task_data_seq->inputs_count.emplace_back(1);
  task_data_seq->inputs_count.emplace_back(1);  // Integral dimension info
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(answer_ptr));
  task_data_seq->outputs_count.emplace_back(1);

  // Create Task
  makhov_m_monte_carlo_method_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  uint8_t *answer_data = task_data_seq->outputs[0];
  double retrieved_value = NAN;
  std::memcpy(&retrieved_value, answer_data, sizeof(double));
  double truncated_value = std::round(retrieved_value * 100) / 100;
  EXPECT_EQ(reference, truncated_value);
}

TEST(makhov_m_monte_carlo_method_seq, func_is_lnx) {
  // Create data

  std::function<double(const std::vector<double> &)> f = [](const std::vector<double> &x) { return std::log(x[0]); };
  int num_samples = 1000000;
  std::vector<std::pair<double, double>> limits = {{1.0, std::numbers::e}};
  double *answer_ptr = nullptr;
  double reference = 1.0;

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.push_back(reinterpret_cast<uint8_t *>(&f));
  task_data_seq->inputs.push_back(reinterpret_cast<uint8_t *>(&num_samples));
  task_data_seq->inputs.push_back(reinterpret_cast<uint8_t *>(limits.data()));
  task_data_seq->inputs_count.emplace_back(1);
  task_data_seq->inputs_count.emplace_back(1);
  task_data_seq->inputs_count.emplace_back(1);  // Integral dimension info
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(answer_ptr));
  task_data_seq->outputs_count.emplace_back(1);

  // Create Task
  makhov_m_monte_carlo_method_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  uint8_t *answer_data = task_data_seq->outputs[0];
  double retrieved_value = NAN;
  std::memcpy(&retrieved_value, answer_data, sizeof(double));
  double truncated_value = std::round(retrieved_value * 100) / 100;
  EXPECT_EQ(reference, truncated_value);
}

TEST(makhov_m_monte_carlo_method_seq, func_is_x_plus_y) {
  // Create data

  std::function<double(const std::vector<double> &)> f = [](const std::vector<double> &x) { return x[0] + x[1]; };
  int num_samples = 1000000;
  std::vector<std::pair<double, double>> limits = {{0.0, 1.0}, {0.0, 1.0}};
  double *answer_ptr = nullptr;
  double reference = 1.0;

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.push_back(reinterpret_cast<uint8_t *>(&f));
  task_data_seq->inputs.push_back(reinterpret_cast<uint8_t *>(&num_samples));
  task_data_seq->inputs.push_back(reinterpret_cast<uint8_t *>(limits.data()));
  task_data_seq->inputs_count.emplace_back(1);
  task_data_seq->inputs_count.emplace_back(1);
  task_data_seq->inputs_count.emplace_back(2);  // Integral dimension info
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(answer_ptr));
  task_data_seq->outputs_count.emplace_back(1);

  // Create Task
  makhov_m_monte_carlo_method_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  uint8_t *answer_data = task_data_seq->outputs[0];
  double retrieved_value = NAN;
  std::memcpy(&retrieved_value, answer_data, sizeof(double));
  double truncated_value = std::round(retrieved_value * 100) / 100;
  EXPECT_EQ(reference, truncated_value);
}

TEST(makhov_m_monte_carlo_method_seq, func_is_xx_plus_yy) {
  // Create data

  std::function<double(const std::vector<double> &)> f = [](const std::vector<double> &x) {
    return (x[0] * x[0]) + (x[1] * x[1]);
  };
  int num_samples = 1000000;
  std::vector<std::pair<double, double>> limits = {{0.0, 1.0}, {0.0, 1.0}};
  double *answer_ptr = nullptr;
  double reference = 0.67;

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.push_back(reinterpret_cast<uint8_t *>(&f));
  task_data_seq->inputs.push_back(reinterpret_cast<uint8_t *>(&num_samples));
  task_data_seq->inputs.push_back(reinterpret_cast<uint8_t *>(limits.data()));
  task_data_seq->inputs_count.emplace_back(1);
  task_data_seq->inputs_count.emplace_back(1);
  task_data_seq->inputs_count.emplace_back(2);  // Integral dimension info
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(answer_ptr));
  task_data_seq->outputs_count.emplace_back(1);

  // Create Task
  makhov_m_monte_carlo_method_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  uint8_t *answer_data = task_data_seq->outputs[0];
  double retrieved_value = NAN;
  std::memcpy(&retrieved_value, answer_data, sizeof(double));
  double truncated_value = std::round(retrieved_value * 100) / 100;
  EXPECT_EQ(reference, truncated_value);
}

TEST(makhov_m_monte_carlo_method_seq, func_is_xx_plus_yy_plus_zz) {
  // Create data

  std::function<double(const std::vector<double> &)> f = [](const std::vector<double> &x) {
    return (x[0] * x[0]) + (x[1] * x[1]) + (x[2] * x[2]);
  };
  int num_samples = 1000000;
  std::vector<std::pair<double, double>> limits = {{0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}};
  double *answer_ptr = nullptr;
  double reference = 1.0;

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.push_back(reinterpret_cast<uint8_t *>(&f));
  task_data_seq->inputs.push_back(reinterpret_cast<uint8_t *>(&num_samples));
  task_data_seq->inputs.push_back(reinterpret_cast<uint8_t *>(limits.data()));
  task_data_seq->inputs_count.emplace_back(1);
  task_data_seq->inputs_count.emplace_back(1);
  task_data_seq->inputs_count.emplace_back(3);  // Integral dimension info
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(answer_ptr));
  task_data_seq->outputs_count.emplace_back(1);

  // Create Task
  makhov_m_monte_carlo_method_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  uint8_t *answer_data = task_data_seq->outputs[0];
  double retrieved_value = NAN;
  std::memcpy(&retrieved_value, answer_data, sizeof(double));
  double truncated_value = std::round(retrieved_value * 100) / 100;
  EXPECT_EQ(reference, truncated_value);
}