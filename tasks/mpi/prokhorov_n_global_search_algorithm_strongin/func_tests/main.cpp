#include <gtest/gtest.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "core/util/include/util.hpp"
#include "mpi/prokhorov_n_global_search_algorithm_strongin/include/ops_mpi.hpp"

TEST(prokhorov_n_global_search_algorithm_strongin_mpi, test_parallel_quadratic_function) {
  double a = -10.0;
  double b = 10.0;
  double eps = 0.0001;
  std::function<double(double*)> f = [](double* x) { return (*x) * (*x); };

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(&a));
  task_data->inputs_count.push_back(1);
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(&b));
  task_data->inputs_count.push_back(1);
  task_data->outputs.push_back(reinterpret_cast<uint8_t*>(new double));
  task_data->outputs_count.push_back(1);

  prokhorov_n_global_search_algorithm_strongin_mpi::TestTaskParallel test_task_mpi(task_data, f);

  ASSERT_EQ(test_task_mpi.Validation(), true);
  test_task_mpi.PreProcessing();
  test_task_mpi.Run();
  test_task_mpi.PostProcessing();

  double result = *reinterpret_cast<double*>(task_data->outputs[0]);
  EXPECT_NEAR(result, 0.0, eps);
}

TEST(prokhorov_n_global_search_algorithm_strongin_mpi, test_parallel_cos_function) {
  double a = 0.0;
  double b = 3.14;
  double eps = 0.0001;
  std::function<double(double*)> f = [](double* x) { return std::cos(*x); };

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(&a));
  task_data->inputs_count.push_back(1);
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(&b));
  task_data->inputs_count.push_back(1);
  task_data->outputs.push_back(reinterpret_cast<uint8_t*>(new double));
  task_data->outputs_count.push_back(1);

  prokhorov_n_global_search_algorithm_strongin_mpi::TestTaskParallel test_task_mpi(task_data, f);

  ASSERT_EQ(test_task_mpi.Validation(), true);
  test_task_mpi.PreProcessing();
  test_task_mpi.Run();
  test_task_mpi.PostProcessing();

  double result = *reinterpret_cast<double*>(task_data->outputs[0]);
  EXPECT_NEAR(result, 3.14, eps);
}

TEST(prokhorov_n_global_search_algorithm_strongin_mpi, test_parallel_sin_function) {
  double a = 0.0;
  double b = 3.14;
  double eps = 0.0001;
  std::function<double(double*)> f = [](double* x) { return std::sin(*x); };

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(&a));
  task_data->inputs_count.push_back(1);
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(&b));
  task_data->inputs_count.push_back(1);
  task_data->outputs.push_back(reinterpret_cast<uint8_t*>(new double));
  task_data->outputs_count.push_back(1);

  prokhorov_n_global_search_algorithm_strongin_mpi::TestTaskParallel test_task_mpi(task_data, f);

  ASSERT_EQ(test_task_mpi.Validation(), true);
  test_task_mpi.PreProcessing();
  test_task_mpi.Run();
  test_task_mpi.PostProcessing();

  double result = *reinterpret_cast<double*>(task_data->outputs[0]);
  EXPECT_NEAR(result, 0.0, eps);
}

TEST(prokhorov_n_global_search_algorithm_strongin_mpi, test_parallel_abs_function) {
  double a = -5.0;
  double b = 5.0;
  double eps = 0.0001;
  std::function<double(double*)> f = [](double* x) { return std::abs(*x); };

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(&a));
  task_data->inputs_count.push_back(1);
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(&b));
  task_data->inputs_count.push_back(1);
  task_data->outputs.push_back(reinterpret_cast<uint8_t*>(new double));
  task_data->outputs_count.push_back(1);

  prokhorov_n_global_search_algorithm_strongin_mpi::TestTaskParallel test_task_mpi(task_data, f);

  ASSERT_EQ(test_task_mpi.Validation(), true);
  test_task_mpi.PreProcessing();
  test_task_mpi.Run();
  test_task_mpi.PostProcessing();

  double result = *reinterpret_cast<double*>(task_data->outputs[0]);
  EXPECT_NEAR(result, 0.0, eps);
}

TEST(prokhorov_n_global_search_algorithm_strongin_mpi, test_parallel_polynomial_function) {
  double a = -2.0;
  double b = 2.0;
  double eps = 0.0001;
  std::function<double(double*)> f = [](double* x) { return (*x) * (*x) * (*x) - 3 * (*x) * (*x) + 2; };

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(&a));
  task_data->inputs_count.push_back(1);
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(&b));
  task_data->inputs_count.push_back(1);
  task_data->outputs.push_back(reinterpret_cast<uint8_t*>(new double));
  task_data->outputs_count.push_back(1);

  prokhorov_n_global_search_algorithm_strongin_mpi::TestTaskParallel test_task_mpi(task_data, f);

  ASSERT_EQ(test_task_mpi.Validation(), true);
  test_task_mpi.PreProcessing();
  test_task_mpi.Run();
  test_task_mpi.PostProcessing();

  double result = *reinterpret_cast<double*>(task_data->outputs[0]);
  EXPECT_NEAR(result, -2.0, eps);
}