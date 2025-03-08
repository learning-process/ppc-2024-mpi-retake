#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "mpi/prokhorov_n_global_search_algorithm_strongin/include/ops_mpi.hpp"

TEST(prokhorov_n_global_search_algorithm_strongin_mpi, x_polynome) {
  boost::mpi::communicator world;
  double a = -10;
  double b = 0;
  double result = 0;
  const std::function<double(double *)> f = [](const double *x) {
    return ((*x) * (*x) * (*x) * (-0.2465)) + ((*x) * (*x) * (-0.3147)) + 1.0;
  };
  double eps = 0.1;
  double answer = -0.8;

  std::shared_ptr<ppc::core::TaskData> task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(&a));
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(&b));
    task_data_mpi->inputs_count.emplace_back(1);
    task_data_mpi->inputs_count.emplace_back(1);
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(&result));
    task_data_mpi->outputs_count.emplace_back(1);
  }

  prokhorov_n_global_search_algorithm_strongin_mpi::TestTaskParallel test_mpi_task_mpi(task_data_mpi, f);
  ASSERT_EQ(test_mpi_task_mpi.ValidationImpl(), true);
  test_mpi_task_mpi.PreProcessingImpl();
  test_mpi_task_mpi.RunImpl();
  test_mpi_task_mpi.PostProcessingImpl();

  if (world.rank() == 0) {
    EXPECT_NEAR(answer, result, eps);
  }
}

TEST(prokhorov_n_global_search_algorithm_strongin_mpi, quadratic_function) {
  boost::mpi::communicator world;
  double a = -2.0;
  double b = 5.0;
  double result = 0;
  const std::function<double(double *)> f = [](const double *x) { return ((*x) * (*x)) - (4 * (*x)) + 4; };
  double eps = 0.1;
  double answer = 2.0;

  std::shared_ptr<ppc::core::TaskData> task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(&a));
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(&b));
    task_data_mpi->inputs_count.emplace_back(1);
    task_data_mpi->inputs_count.emplace_back(1);
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(&result));
    task_data_mpi->outputs_count.emplace_back(1);
  }

  prokhorov_n_global_search_algorithm_strongin_mpi::TestTaskParallel test_mpi_task_mpi(task_data_mpi, f);
  ASSERT_EQ(test_mpi_task_mpi.ValidationImpl(), true);
  test_mpi_task_mpi.PreProcessingImpl();
  test_mpi_task_mpi.RunImpl();
  test_mpi_task_mpi.PostProcessingImpl();

  if (world.rank() == 0) {
    EXPECT_NEAR(answer, result, eps);
  }
}

TEST(prokhorov_n_global_search_algorithm_strongin_mpi, logarithmic_function) {
  boost::mpi::communicator world;
  double a = 0.0;
  double b = 10.0;
  double result = 0;
  const std::function<double(double *)> f = [](const double *x) { return std::log(*x + 1); };
  double eps = 0.1;
  double answer = 0.0;

  std::shared_ptr<ppc::core::TaskData> task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(&a));
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(&b));
    task_data_mpi->inputs_count.emplace_back(1);
    task_data_mpi->inputs_count.emplace_back(1);
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(&result));
    task_data_mpi->outputs_count.emplace_back(1);
  }

  prokhorov_n_global_search_algorithm_strongin_mpi::TestTaskParallel test_mpi_task_mpi(task_data_mpi, f);
  ASSERT_EQ(test_mpi_task_mpi.ValidationImpl(), true);
  test_mpi_task_mpi.PreProcessingImpl();
  test_mpi_task_mpi.RunImpl();
  test_mpi_task_mpi.PostProcessingImpl();

  if (world.rank() == 0) {
    EXPECT_NEAR(answer, result, eps);
  }
}

TEST(prokhorov_n_global_search_algorithm_strongin_mpi, absolute_value_function) {
  boost::mpi::communicator world;
  double a = -3.0;
  double b = 3.0;
  double result = 0;
  const std::function<double(double *)> f = [](const double *x) { return std::abs(*x); };
  double eps = 0.1;
  double answer = 0.0;

  std::shared_ptr<ppc::core::TaskData> task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(&a));
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(&b));
    task_data_mpi->inputs_count.emplace_back(1);
    task_data_mpi->inputs_count.emplace_back(1);
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(&result));
    task_data_mpi->outputs_count.emplace_back(1);
  }

  prokhorov_n_global_search_algorithm_strongin_mpi::TestTaskParallel test_mpi_task_mpi(task_data_mpi, f);
  ASSERT_EQ(test_mpi_task_mpi.ValidationImpl(), true);
  test_mpi_task_mpi.PreProcessingImpl();
  test_mpi_task_mpi.RunImpl();
  test_mpi_task_mpi.PostProcessingImpl();

  if (world.rank() == 0) {
    EXPECT_NEAR(answer, result, eps);
  }
}

TEST(prokhorov_n_global_search_algorithm_strongin_mpi, x_polynome_2) {
  boost::mpi::communicator world;
  double a = -10;
  double b = 0;
  double result = 0;
  const std::function<double(double *)> f = [](const double *x) {
    return ((*x) * (*x) * (*x) * (-0.2465)) + ((*x) * (*x) * (-0.3147)) + 1.0;
  };
  double eps = 0.1;
  double answer = -0.8;

  std::shared_ptr<ppc::core::TaskData> task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(&a));
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(&b));
    task_data_mpi->inputs_count.emplace_back(1);
    task_data_mpi->inputs_count.emplace_back(1);
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(&result));
    task_data_mpi->outputs_count.emplace_back(1);
  }

  prokhorov_n_global_search_algorithm_strongin_mpi::TestTaskParallel test_mpi_task_mpi(task_data_mpi, f);
  ASSERT_EQ(test_mpi_task_mpi.ValidationImpl(), true);
  test_mpi_task_mpi.PreProcessingImpl();
  test_mpi_task_mpi.RunImpl();
  test_mpi_task_mpi.PostProcessingImpl();

  if (world.rank() == 0) {
    EXPECT_NEAR(answer, result, eps);
  }
}

TEST(prokhorov_n_global_search_algorithm_strongin_mpi, exp_function) {
  boost::mpi::communicator world;
  double a = -1;
  double b = 5.389;
  double result = 0;
  const std::function<double(double *)> f = [](const double *x) { return std::exp(*x) - 0.13645; };
  double eps = 0.1;
  double answer = -1.0;

  std::shared_ptr<ppc::core::TaskData> task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(&a));
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(&b));
    task_data_mpi->inputs_count.emplace_back(1);
    task_data_mpi->inputs_count.emplace_back(1);
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(&result));
    task_data_mpi->outputs_count.emplace_back(1);
  }

  prokhorov_n_global_search_algorithm_strongin_mpi::TestTaskParallel test_mpi_task_mpi(task_data_mpi, f);
  ASSERT_EQ(test_mpi_task_mpi.ValidationImpl(), true);
  test_mpi_task_mpi.PreProcessingImpl();
  test_mpi_task_mpi.RunImpl();
  test_mpi_task_mpi.PostProcessingImpl();

  if (world.rank() == 0) {
    EXPECT_NEAR(answer, result, eps);
  }
}
