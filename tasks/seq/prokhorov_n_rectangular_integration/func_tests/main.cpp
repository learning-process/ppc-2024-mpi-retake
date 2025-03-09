#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <memory>
#include <numbers>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/prokhorov_n_rectangular_integration/include/ops_seq.hpp"

TEST(prokhorov_n_rectangular_integration_seq, test_integration_cos_x) {
  const double lower_bound = 0.0;
  const double upper_bound = std::numbers::pi / 2.0;
  const int n = 1000;
  const double expected_result = 1.0;

  std::vector<double> in = {lower_bound, upper_bound, static_cast<double>(n)};
  std::vector<double> out(1, 0.0);

  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  auto test_task_sequential =
      std::make_shared<prokhorov_n_rectangular_integration_seq::TestTaskSequential>(task_data_seq);
  test_task_sequential->SetFunction([](double x) { return std::cos(x); });

  ASSERT_EQ(test_task_sequential->ValidationImpl(), true);
  test_task_sequential->PreProcessingImpl();
  test_task_sequential->RunImpl();
  test_task_sequential->PostProcessingImpl();

  ASSERT_NEAR(out[0], expected_result, 1e-3);
}

TEST(prokhorov_n_rectangular_integration_seq, test_integration_x_cubed) {
  const double lower_bound = 0.0;
  const double upper_bound = 1.0;
  const int n = 1000;
  const double expected_result = 0.25;

  std::vector<double> in = {lower_bound, upper_bound, static_cast<double>(n)};
  std::vector<double> out(1, 0.0);

  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  auto test_task_sequential =
      std::make_shared<prokhorov_n_rectangular_integration_seq::TestTaskSequential>(task_data_seq);
  test_task_sequential->SetFunction([](double x) { return x * x * x; });

  ASSERT_EQ(test_task_sequential->ValidationImpl(), true);
  test_task_sequential->PreProcessingImpl();
  test_task_sequential->RunImpl();
  test_task_sequential->PostProcessingImpl();

  ASSERT_NEAR(out[0], expected_result, 1e-3);
}

TEST(prokhorov_n_rectangular_integration_seq, test_integration_sqrt_x) {
  const double lower_bound = 0.0;
  const double upper_bound = 1.0;
  const int n = 1000;
  const double expected_result = 2.0 / 3.0;

  std::vector<double> in = {lower_bound, upper_bound, static_cast<double>(n)};
  std::vector<double> out(1, 0.0);

  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  auto test_task_sequential =
      std::make_shared<prokhorov_n_rectangular_integration_seq::TestTaskSequential>(task_data_seq);
  test_task_sequential->SetFunction([](double x) { return std::sqrt(x); });

  ASSERT_EQ(test_task_sequential->ValidationImpl(), true);
  test_task_sequential->PreProcessingImpl();
  test_task_sequential->RunImpl();
  test_task_sequential->PostProcessingImpl();

  ASSERT_NEAR(out[0], expected_result, 1e-3);
}

TEST(prokhorov_n_rectangular_integration_seq, test_integration_one_over_x) {
  const double lower_bound = 1.0;
  const double upper_bound = 2.0;
  const int n = 1000;
  const double expected_result = std::numbers::ln2;

  std::vector<double> in = {lower_bound, upper_bound, static_cast<double>(n)};
  std::vector<double> out(1, 0.0);

  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  auto test_task_sequential =
      std::make_shared<prokhorov_n_rectangular_integration_seq::TestTaskSequential>(task_data_seq);
  test_task_sequential->SetFunction([](double x) { return 1.0 / x; });

  ASSERT_EQ(test_task_sequential->ValidationImpl(), true);
  test_task_sequential->PreProcessingImpl();
  test_task_sequential->RunImpl();
  test_task_sequential->PostProcessingImpl();

  ASSERT_NEAR(out[0], expected_result, 1e-3);
}

TEST(prokhorov_n_rectangular_integration_seq, test_integration_sin_squared_x) {
  const double lower_bound = 0.0;
  const double upper_bound = std::numbers::pi;
  const int n = 1000;
  const double expected_result = std::numbers::pi / 2.0;

  std::vector<double> in = {lower_bound, upper_bound, static_cast<double>(n)};
  std::vector<double> out(1, 0.0);

  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  auto test_task_sequential =
      std::make_shared<prokhorov_n_rectangular_integration_seq::TestTaskSequential>(task_data_seq);
  test_task_sequential->SetFunction([](double x) { return std::sin(x) * std::sin(x); });

  ASSERT_EQ(test_task_sequential->ValidationImpl(), true);
  test_task_sequential->PreProcessingImpl();
  test_task_sequential->RunImpl();
  test_task_sequential->PostProcessingImpl();

  ASSERT_NEAR(out[0], expected_result, 1e-3);
}

TEST(prokhorov_n_rectangular_integration_seq, test_integration_exp_minus_x_squared) {
  const double lower_bound = 0.0;
  const double upper_bound = 1.0;
  const int n = 1000;
  const double expected_result = 0.746824;

  std::vector<double> in = {lower_bound, upper_bound, static_cast<double>(n)};
  std::vector<double> out(1, 0.0);

  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  auto test_task_sequential =
      std::make_shared<prokhorov_n_rectangular_integration_seq::TestTaskSequential>(task_data_seq);
  test_task_sequential->SetFunction([](double x) { return std::exp(-x * x); });

  ASSERT_EQ(test_task_sequential->ValidationImpl(), true);
  test_task_sequential->PreProcessingImpl();
  test_task_sequential->RunImpl();
  test_task_sequential->PostProcessingImpl();

  ASSERT_NEAR(out[0], expected_result, 1e-3);
}

TEST(prokhorov_n_rectangular_integration_seq, test_integration_log_x) {
  const double lower_bound = 1.0;
  const double upper_bound = 2.0;
  const int n = 1000;
  const double expected_result = (2.0 * std::numbers::ln2) - 1.0;

  std::vector<double> in = {lower_bound, upper_bound, static_cast<double>(n)};
  std::vector<double> out(1, 0.0);

  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  auto test_task_sequential =
      std::make_shared<prokhorov_n_rectangular_integration_seq::TestTaskSequential>(task_data_seq);
  test_task_sequential->SetFunction([](double x) { return std::log(x); });

  ASSERT_EQ(test_task_sequential->ValidationImpl(), true);
  test_task_sequential->PreProcessingImpl();
  test_task_sequential->RunImpl();
  test_task_sequential->PostProcessingImpl();

  ASSERT_NEAR(out[0], expected_result, 1e-3);
}

TEST(prokhorov_n_rectangular_integration_seq, test_integration_x_sin_x) {
  const double lower_bound = 0.0;
  const double upper_bound = std::numbers::pi;
  const int n = 1000;
  const double expected_result = std::numbers::pi;

  std::vector<double> in = {lower_bound, upper_bound, static_cast<double>(n)};
  std::vector<double> out(1, 0.0);

  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  auto test_task_sequential =
      std::make_shared<prokhorov_n_rectangular_integration_seq::TestTaskSequential>(task_data_seq);
  test_task_sequential->SetFunction([](double x) { return x * std::sin(x); });

  ASSERT_EQ(test_task_sequential->ValidationImpl(), true);
  test_task_sequential->PreProcessingImpl();
  test_task_sequential->RunImpl();
  test_task_sequential->PostProcessingImpl();

  ASSERT_NEAR(out[0], expected_result, 1e-3);
}

TEST(prokhorov_n_rectangular_integration_seq, test_integration_atan_x) {
  const double lower_bound = 0.0;
  const double upper_bound = 1.0;
  const int n = 1000;
  const double expected_result = (std::numbers::pi / 4.0) - (0.5 * std::numbers::ln2);

  std::vector<double> in = {lower_bound, upper_bound, static_cast<double>(n)};
  std::vector<double> out(1, 0.0);

  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  auto test_task_sequential =
      std::make_shared<prokhorov_n_rectangular_integration_seq::TestTaskSequential>(task_data_seq);
  test_task_sequential->SetFunction([](double x) { return std::atan(x); });

  ASSERT_EQ(test_task_sequential->ValidationImpl(), true);
  test_task_sequential->PreProcessingImpl();
  test_task_sequential->RunImpl();
  test_task_sequential->PostProcessingImpl();

  ASSERT_NEAR(out[0], expected_result, 1e-3);
}