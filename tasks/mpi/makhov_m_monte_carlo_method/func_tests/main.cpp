// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"
#include "mpi/makhov_m_monte_carlo_method/include/ops_mpi.hpp"

TEST(makhov_m_monte_carlo_method, func_is_x_pow2) {
  // Create data
  boost::mpi::communicator world;
  std::string f = "x*x";
  int num_samples = 1000000;
  std::vector<std::pair<double, double>> limits = {{0.0, 1.0}};
  double *answer_ptr = nullptr;
  double reference = 0.33;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_par->inputs.push_back(reinterpret_cast<uint8_t *>(&f));
    task_data_par->inputs.push_back(reinterpret_cast<uint8_t *>(&num_samples));
    task_data_par->inputs.push_back(reinterpret_cast<uint8_t *>(limits.data()));
    task_data_par->inputs_count.emplace_back(1);
    task_data_par->inputs_count.emplace_back(1);
    task_data_par->inputs_count.emplace_back(1);  // Integral dimension info
    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t *>(answer_ptr));
    task_data_par->outputs_count.emplace_back(1);
  }

  // Create Task
  makhov_m_monte_carlo_method_mpi::TestMPITaskParallel test_mpi_task_parallel(task_data_par);
  ASSERT_TRUE(test_mpi_task_parallel.ValidationImpl());
  test_mpi_task_parallel.PreProcessingImpl();
  test_mpi_task_parallel.RunImpl();
  test_mpi_task_parallel.PostProcessingImpl();

  if (world.rank() == 0) {
    uint8_t *answer_data = task_data_par->outputs[0];
    double retrieved_value = NAN;
    std::memcpy(&retrieved_value, answer_data, sizeof(double));
    EXPECT_NEAR(reference, retrieved_value, 0.01);
  }
}

TEST(makhov_m_monte_carlo_method, func_is_x_plus_y) {
  // Create data
  boost::mpi::communicator world;
  std::string f = "x+y";
  int num_samples = 1000000;
  std::vector<std::pair<double, double>> limits = {{0.0, 1.0}, {0.0, 1.0}};
  double *answer_ptr = nullptr;
  double reference = 1.0;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_par->inputs.push_back(reinterpret_cast<uint8_t *>(&f));
    task_data_par->inputs.push_back(reinterpret_cast<uint8_t *>(&num_samples));
    task_data_par->inputs.push_back(reinterpret_cast<uint8_t *>(limits.data()));
    task_data_par->inputs_count.emplace_back(1);
    task_data_par->inputs_count.emplace_back(1);
    task_data_par->inputs_count.emplace_back(2);  // Integral dimension info
    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t *>(answer_ptr));
    task_data_par->outputs_count.emplace_back(1);
  }

  // Create Task
  makhov_m_monte_carlo_method_mpi::TestMPITaskParallel test_mpi_task_parallel(task_data_par);
  ASSERT_TRUE(test_mpi_task_parallel.ValidationImpl());
  test_mpi_task_parallel.PreProcessingImpl();
  test_mpi_task_parallel.RunImpl();
  test_mpi_task_parallel.PostProcessingImpl();

  if (world.rank() == 0) {
    uint8_t *answer_data = task_data_par->outputs[0];
    double retrieved_value = NAN;
    std::memcpy(&retrieved_value, answer_data, sizeof(double));
    EXPECT_NEAR(reference, retrieved_value, 0.01);
  }
}

TEST(makhov_m_monte_carlo_method, func_is_xx_plus_yy) {
  // Create data
  boost::mpi::communicator world;
  std::string f = "x*x+y*y";
  int num_samples = 1000000;
  std::vector<std::pair<double, double>> limits = {{0.0, 1.0}, {0.0, 1.0}};
  double *answer_ptr = nullptr;
  double reference = 0.67;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_par->inputs.push_back(reinterpret_cast<uint8_t *>(&f));
    task_data_par->inputs.push_back(reinterpret_cast<uint8_t *>(&num_samples));
    task_data_par->inputs.push_back(reinterpret_cast<uint8_t *>(limits.data()));
    task_data_par->inputs_count.emplace_back(1);
    task_data_par->inputs_count.emplace_back(1);
    task_data_par->inputs_count.emplace_back(2);  // Integral dimension info
    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t *>(answer_ptr));
    task_data_par->outputs_count.emplace_back(1);
  }

  // Create Task
  makhov_m_monte_carlo_method_mpi::TestMPITaskParallel test_mpi_task_parallel(task_data_par);
  ASSERT_TRUE(test_mpi_task_parallel.ValidationImpl());
  test_mpi_task_parallel.PreProcessingImpl();
  test_mpi_task_parallel.RunImpl();
  test_mpi_task_parallel.PostProcessingImpl();

  if (world.rank() == 0) {
    uint8_t *answer_data = task_data_par->outputs[0];
    double retrieved_value = NAN;
    std::memcpy(&retrieved_value, answer_data, sizeof(double));
    EXPECT_NEAR(reference, retrieved_value, 0.01);
  }
}

TEST(makhov_m_monte_carlo_method, func_is_xx_plus_yy_plus_zz) {
  // Create data
  boost::mpi::communicator world;
  std::string f = "x*x+y*y+z*z";
  int num_samples = 1000000;
  std::vector<std::pair<double, double>> limits = {{0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}};
  double *answer_ptr = nullptr;
  double reference = 1.0;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_par->inputs.push_back(reinterpret_cast<uint8_t *>(&f));
    task_data_par->inputs.push_back(reinterpret_cast<uint8_t *>(&num_samples));
    task_data_par->inputs.push_back(reinterpret_cast<uint8_t *>(limits.data()));
    task_data_par->inputs_count.emplace_back(1);
    task_data_par->inputs_count.emplace_back(1);
    task_data_par->inputs_count.emplace_back(3);  // Integral dimension info
    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t *>(answer_ptr));
    task_data_par->outputs_count.emplace_back(1);
  }

  // Create Task
  makhov_m_monte_carlo_method_mpi::TestMPITaskParallel test_mpi_task_parallel(task_data_par);
  ASSERT_TRUE(test_mpi_task_parallel.ValidationImpl());
  test_mpi_task_parallel.PreProcessingImpl();
  test_mpi_task_parallel.RunImpl();
  test_mpi_task_parallel.PostProcessingImpl();

  if (world.rank() == 0) {
    uint8_t *answer_data = task_data_par->outputs[0];
    double retrieved_value = NAN;
    std::memcpy(&retrieved_value, answer_data, sizeof(double));
    EXPECT_NEAR(reference, retrieved_value, 0.01);
  }
}