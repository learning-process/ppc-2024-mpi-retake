#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/prokhorov_n_global_search_algorithm_strongin/include/ops_seq.hpp"

TEST(prokhorov_n_global_search_algorithm_strongin_seq, Test_Strongin_Algorithm_Quadratic_Function) {
  std::vector<double> in_a = {-10.0};
  std::vector<double> in_b = {10.0};
  std::vector<double> in_epsilon = {0.001};
  std::vector<double> out(1, 0.0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_a.data()));
  task_data_seq->inputs_count.emplace_back(in_a.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_b.data()));
  task_data_seq->inputs_count.emplace_back(in_b.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_epsilon.data()));
  task_data_seq->inputs_count.emplace_back(in_epsilon.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  auto quadratic_function = [](double x) { return x * x; };
  prokhorov_n_global_search_algorithm_strongin_seq::TestTaskSequential test_task_sequential(task_data_seq,
                                                                                            quadratic_function);
  ASSERT_EQ(test_task_sequential.ValidationImpl(), true);
  test_task_sequential.PreProcessingImpl();
  test_task_sequential.RunImpl();
  test_task_sequential.PostProcessingImpl();

  EXPECT_NEAR(out[0], 0.0, 0.001);
}

TEST(prokhorov_n_global_search_algorithm_strongin_seq, Test_Strongin_Algorithm_Absolute_Function) {
  std::vector<double> in_a = {-5.0};
  std::vector<double> in_b = {5.0};
  std::vector<double> in_epsilon = {0.001};
  std::vector<double> out(1, 0.0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_a.data()));
  task_data_seq->inputs_count.emplace_back(in_a.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_b.data()));
  task_data_seq->inputs_count.emplace_back(in_b.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_epsilon.data()));
  task_data_seq->inputs_count.emplace_back(in_epsilon.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  auto absolute_function = [](double x) { return std::abs(x); };
  prokhorov_n_global_search_algorithm_strongin_seq::TestTaskSequential test_task_sequential(task_data_seq,
                                                                                            absolute_function);
  ASSERT_EQ(test_task_sequential.ValidationImpl(), true);
  test_task_sequential.PreProcessingImpl();
  test_task_sequential.RunImpl();
  test_task_sequential.PostProcessingImpl();

  EXPECT_NEAR(out[0], 0.0, 0.001);
}

TEST(prokhorov_n_global_search_algorithm_strongin_seq, Test_Strongin_Algorithm_SquareRoot_Function) {
  std::vector<double> in_a = {0.0};
  std::vector<double> in_b = {10.0};
  std::vector<double> in_epsilon = {0.001};
  std::vector<double> out(1, 0.0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_a.data()));
  task_data_seq->inputs_count.emplace_back(in_a.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_b.data()));
  task_data_seq->inputs_count.emplace_back(in_b.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_epsilon.data()));
  task_data_seq->inputs_count.emplace_back(in_epsilon.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  auto square_root_function = [](double x) { return std::sqrt(x); };
  prokhorov_n_global_search_algorithm_strongin_seq::TestTaskSequential test_task_sequential(task_data_seq,
                                                                                            square_root_function);
  ASSERT_EQ(test_task_sequential.ValidationImpl(), true);
  test_task_sequential.PreProcessingImpl();
  test_task_sequential.RunImpl();
  test_task_sequential.PostProcessingImpl();

  EXPECT_NEAR(out[0], 0.0, 0.001);
}

TEST(prokhorov_n_global_search_algorithm_strongin_seq, Test_Strongin_Algorithm_Logarithmic_Function) {
  std::vector<double> in_a = {0.1};
  std::vector<double> in_b = {10.0};
  std::vector<double> in_epsilon = {0.001};
  std::vector<double> out(1, 0.0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_a.data()));
  task_data_seq->inputs_count.emplace_back(in_a.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_b.data()));
  task_data_seq->inputs_count.emplace_back(in_b.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_epsilon.data()));
  task_data_seq->inputs_count.emplace_back(in_epsilon.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  auto logarithmic_function = [](double x) { return std::log(x); };
  prokhorov_n_global_search_algorithm_strongin_seq::TestTaskSequential test_task_sequential(task_data_seq,
                                                                                            logarithmic_function);
  ASSERT_EQ(test_task_sequential.ValidationImpl(), true);
  test_task_sequential.PreProcessingImpl();
  test_task_sequential.RunImpl();
  test_task_sequential.PostProcessingImpl();

  EXPECT_NEAR(out[0], 0.1, 0.001);  // ќжидаем x = 0.1
}