// Copyright 2024 Kabalova Valeria
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>
#include <numbers>
#include <random>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"
#include "mpi/kabalova_v_strongin/include/strongin.h"

namespace kabalova_v_strongin_mpi {

std::pair<double, double> GenerateBounds(double left, double right) {
  // int seed = 101;
  std::random_device seed;
  std::mt19937 gen(seed());
  std::uniform_real_distribution distrib(left, right);
  std::pair<double, double> tmp;
  tmp.first = distrib(gen);
  tmp.second = left;
  while (tmp.second < tmp.first) {
    tmp.second = distrib(gen);
  }
  return tmp;
}
}  // namespace kabalova_v_strongin_mpi

// Все минимумы найдены +- по графику.

TEST(kabalova_v_strongin_mpi, x_square) {
  double left = 0;
  double right = 10;
  double res1 = 0;
  double res2 = 0;
  const std::function<double(double *)> f = [](const double *x) { return *x * *x; };
  double answer1 = 0;
  double answer2 = f(&answer1);
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
  kabalova_v_strongin_mpi::TestMPITaskSequential test_mpi_task_seq(task_data_seq, f);
  ASSERT_EQ(test_mpi_task_seq.ValidationImpl(), true);
  test_mpi_task_seq.PreProcessingImpl();
  test_mpi_task_seq.RunImpl();
  test_mpi_task_seq.PostProcessingImpl();
  EXPECT_NEAR(answer1, res1, eps);
  EXPECT_NEAR(answer2, res2, eps);
}

TEST(kabalova_v_strongin_mpi, sin) {
  double left = -std::numbers::pi;
  double right = std::numbers::pi;
  double res1 = 0;
  double res2 = 0;
  const std::function<double(double *)> f = [](const double *x) { return std::sin(*x); };
  double eps = 0.1;
  double answer1 = -std::numbers::pi / 2;
  double answer2 = f(&answer1);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&left));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&right));
  task_data_seq->inputs_count.emplace_back(2);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&res1));
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&res2));
  task_data_seq->outputs_count.emplace_back(2);

  // Create Task
  kabalova_v_strongin_mpi::TestMPITaskSequential test_mpi_task_seq(task_data_seq, f);
  ASSERT_EQ(test_mpi_task_seq.ValidationImpl(), true);
  test_mpi_task_seq.PreProcessingImpl();
  test_mpi_task_seq.RunImpl();
  test_mpi_task_seq.PostProcessingImpl();
  EXPECT_NEAR(answer1, res1, eps);
  EXPECT_NEAR(answer2, res2, eps);
}

TEST(kabalova_v_strongin_mpi, cos) {
  double left = -std::numbers::pi;
  double right = std::numbers::pi / 2;
  double res1 = 0;
  double res2 = 0;
  const std::function<double(double *)> f = [](const double *x) { return std::cos(*x); };
  double eps = 0.1;
  double answer1 = -std::numbers::pi;
  double answer2 = f(&answer1);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&left));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&right));
  task_data_seq->inputs_count.emplace_back(2);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&res1));
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&res2));
  task_data_seq->outputs_count.emplace_back(2);

  // Create Task
  kabalova_v_strongin_mpi::TestMPITaskSequential test_mpi_task_seq(task_data_seq, f);
  ASSERT_EQ(test_mpi_task_seq.ValidationImpl(), true);
  test_mpi_task_seq.PreProcessingImpl();
  test_mpi_task_seq.RunImpl();
  test_mpi_task_seq.PostProcessingImpl();
  EXPECT_NEAR(answer1, res1, eps);
  EXPECT_NEAR(answer2, res2, eps);
}

TEST(kabalova_v_strongin_mpi, x_polynome) {
  double left = -10;
  double right = 0;
  double res1 = 0;
  double res2 = 0;
  const std::function<double(double *)> f = [](const double *x) {
    return (*x * *x * *x * (-0.2465)) + (*x * *x * (-0.3147)) + 1.0;
  };
  double eps = 0.1;
  double answer1 = -0.8;
  double answer2 = f(&answer1);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&left));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&right));
  task_data_seq->inputs_count.emplace_back(2);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&res1));
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&res2));
  task_data_seq->outputs_count.emplace_back(2);

  // Create Task
  kabalova_v_strongin_mpi::TestMPITaskSequential test_mpi_task_seq(task_data_seq, f);
  ASSERT_EQ(test_mpi_task_seq.ValidationImpl(), true);
  test_mpi_task_seq.PreProcessingImpl();
  test_mpi_task_seq.RunImpl();
  test_mpi_task_seq.PostProcessingImpl();
  EXPECT_NEAR(answer1, res1, eps);
  EXPECT_NEAR(answer2, res2, eps);
}

TEST(kabalova_v_strongin_mpi, x_polynome2) {
  boost::mpi::communicator world;
  double left = -10;
  double right = 0;
  double res1 = 0;
  double res2 = 0;
  const std::function<double(double *)> f = [](const double *x) {
    return (*x * *x * *x * (-0.2465)) + (*x * *x * (-0.3147)) + 1.0;
  };
  double eps = 0.1;
  double answer1 = -0.8;
  double answer2 = f(&answer1);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(&left));
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(&right));
    task_data_mpi->inputs_count.emplace_back(2);
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(&res1));
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(&res2));
    task_data_mpi->outputs_count.emplace_back(2);
  }
  // Create Task
  kabalova_v_strongin_mpi::TestMPITaskParallel test_mpi_task_mpi(task_data_mpi, f);
  ASSERT_EQ(test_mpi_task_mpi.ValidationImpl(), true);
  test_mpi_task_mpi.PreProcessingImpl();
  test_mpi_task_mpi.RunImpl();
  test_mpi_task_mpi.PostProcessingImpl();
  if (world.rank() == 0) {
    EXPECT_NEAR(answer1, res1, eps);
    EXPECT_NEAR(answer2, res2, eps);
  }
}

TEST(kabalova_v_strongin_mpi, x_square_mpi_and_seq) {
  std::pair<double, double> tmp = kabalova_v_strongin_mpi::GenerateBounds(2, 100);
  boost::mpi::communicator world;
  double left = tmp.first;
  double right = tmp.second;
  double res1 = 0;
  double res2 = 0;
  const std::function<double(double *)> f = [](const double *x) { return *x * *x; };
  double eps = 0.3;
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&left));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&right));
  task_data_seq->inputs_count.emplace_back(2);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&res1));
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&res2));
  task_data_seq->outputs_count.emplace_back(2);

  // Create Task
  kabalova_v_strongin_mpi::TestMPITaskSequential test_mpi_task_seq(task_data_seq, f);
  ASSERT_EQ(test_mpi_task_seq.ValidationImpl(), true);
  test_mpi_task_seq.PreProcessingImpl();
  test_mpi_task_seq.RunImpl();
  test_mpi_task_seq.PostProcessingImpl();
  // Create data
  double res3 = 0;
  double res4 = 0;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(&left));
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(&right));
    task_data_mpi->inputs_count.emplace_back(2);
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(&res3));
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(&res4));
    task_data_mpi->outputs_count.emplace_back(2);
  }
  // Create Task
  kabalova_v_strongin_mpi::TestMPITaskParallel test_mpi_task_mpi(task_data_mpi, f);
  ASSERT_EQ(test_mpi_task_mpi.ValidationImpl(), true);
  test_mpi_task_mpi.PreProcessingImpl();
  test_mpi_task_mpi.RunImpl();
  test_mpi_task_mpi.PostProcessingImpl();

  if (world.rank() == 0) {
    EXPECT_NEAR(res1, res3, eps);
    EXPECT_NEAR(res2, res4, eps);
  }
}

TEST(kabalova_v_strongin_mpi, x_polynome_mpi_and_seq1) {
  boost::mpi::communicator world;
  std::pair<double, double> tmp = kabalova_v_strongin_mpi::GenerateBounds(-1, 1);
  double left = tmp.first;
  double right = tmp.second;
  double res1 = 0;
  double res2 = 0;
  const std::function<double(double *)> f = [](const double *x) {
    return (*x * *x * *x * (-0.1465)) + (*x * *x * (-0.2147)) - 1.0;
  };
  double eps = 0.3;
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&left));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&right));
  task_data_seq->inputs_count.emplace_back(2);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&res1));
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&res2));
  task_data_seq->outputs_count.emplace_back(2);

  // Create Task
  kabalova_v_strongin_mpi::TestMPITaskSequential test_mpi_task_seq(task_data_seq, f);
  ASSERT_EQ(test_mpi_task_seq.ValidationImpl(), true);
  test_mpi_task_seq.PreProcessingImpl();
  test_mpi_task_seq.RunImpl();
  test_mpi_task_seq.PostProcessingImpl();
  // Create data
  double res3 = 0;
  double res4 = 0;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(&left));
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(&right));
    task_data_mpi->inputs_count.emplace_back(2);
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(&res3));
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(&res4));
    task_data_mpi->outputs_count.emplace_back(2);
  }
  // Create Task
  kabalova_v_strongin_mpi::TestMPITaskParallel test_mpi_task_mpi(task_data_mpi, f);
  ASSERT_EQ(test_mpi_task_mpi.ValidationImpl(), true);
  test_mpi_task_mpi.PreProcessingImpl();
  test_mpi_task_mpi.RunImpl();
  test_mpi_task_mpi.PostProcessingImpl();
  if (world.rank() == 0) {
    EXPECT_NEAR(res1, res3, eps);
    EXPECT_NEAR(res2, res4, eps);
  }
}

TEST(kabalova_v_strongin_mpi, x_polynome_mpi_and_seq2) {
  boost::mpi::communicator world;
  std::pair<double, double> tmp = kabalova_v_strongin_mpi::GenerateBounds(0.96, 4.04);
  double left = tmp.first;
  double right = tmp.second;

  double res1 = 0;
  double res2 = 0;
  const std::function<double(double *)> f = [](const double *x) {
    return (*x * *x * *x * (0.109)) + (*x * *x * (0.3147)) + (*x * *x * (-0.574)) + *x + 0.32471;
  };
  double eps = 0.3;
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&left));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&right));
  task_data_seq->inputs_count.emplace_back(2);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&res1));
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&res2));
  task_data_seq->outputs_count.emplace_back(2);

  // Create Task
  kabalova_v_strongin_mpi::TestMPITaskSequential test_mpi_task_seq(task_data_seq, f);
  ASSERT_EQ(test_mpi_task_seq.ValidationImpl(), true);
  test_mpi_task_seq.PreProcessingImpl();
  test_mpi_task_seq.RunImpl();
  test_mpi_task_seq.PostProcessingImpl();
  // Create data
  double res3 = 0;
  double res4 = 0;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(&left));
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(&right));
    task_data_mpi->inputs_count.emplace_back(2);
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(&res3));
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(&res4));
    task_data_mpi->outputs_count.emplace_back(2);
  }
  // Create Task
  kabalova_v_strongin_mpi::TestMPITaskParallel test_mpi_task_mpi(task_data_mpi, f);
  ASSERT_EQ(test_mpi_task_mpi.ValidationImpl(), true);
  test_mpi_task_mpi.PreProcessingImpl();
  test_mpi_task_mpi.RunImpl();
  test_mpi_task_mpi.PostProcessingImpl();
  if (world.rank() == 0) {
    EXPECT_NEAR(res1, res3, eps);
    EXPECT_NEAR(res2, res4, eps);
  }
}

TEST(kabalova_v_strongin_mpi, exp) {
  boost::mpi::communicator world;
  std::pair<double, double> tmp = kabalova_v_strongin_mpi::GenerateBounds(-1, 5.389);
  double left = tmp.first;
  double right = tmp.second;
  double res1 = 0;
  double res2 = 0;
  const std::function<double(double *)> f = [](const double *x) { return std::exp(*x) - 0.13645; };
  double eps = 0.1;
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&left));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&right));
  task_data_seq->inputs_count.emplace_back(2);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&res1));
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&res2));
  task_data_seq->outputs_count.emplace_back(2);

  // Create Task
  kabalova_v_strongin_mpi::TestMPITaskSequential test_mpi_task_seq(task_data_seq, f);
  ASSERT_EQ(test_mpi_task_seq.ValidationImpl(), true);
  test_mpi_task_seq.PreProcessingImpl();
  test_mpi_task_seq.RunImpl();
  test_mpi_task_seq.PostProcessingImpl();
  // Create data
  double res3 = 0;
  double res4 = 0;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(&left));
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(&right));
    task_data_mpi->inputs_count.emplace_back(2);
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(&res3));
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(&res4));
    task_data_mpi->outputs_count.emplace_back(2);
  }
  // Create Task
  kabalova_v_strongin_mpi::TestMPITaskParallel test_mpi_task_mpi(task_data_mpi, f);
  ASSERT_EQ(test_mpi_task_mpi.ValidationImpl(), true);
  test_mpi_task_mpi.PreProcessingImpl();
  test_mpi_task_mpi.RunImpl();
  test_mpi_task_mpi.PostProcessingImpl();
  if (world.rank() == 0) {
    EXPECT_NEAR(res1, res3, eps);
    EXPECT_NEAR(res2, res4, eps);
  }
}

TEST(kabalova_v_strongin_mpi, exp_for_perf_tests) {
  boost::mpi::communicator world;
  double left = -1.0;
  double right = 4.0;
  double res1 = 0;
  double res2 = 0;
  const std::function<double(double *)> f = [](const double *x) { return std::exp(*x) - 1.0; };
  double eps = 0.1;
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&left));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&right));
  task_data_seq->inputs_count.emplace_back(2);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&res1));
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&res2));
  task_data_seq->outputs_count.emplace_back(2);

  // Create Task
  kabalova_v_strongin_mpi::TestMPITaskSequential test_mpi_task_seq(task_data_seq, f);
  ASSERT_EQ(test_mpi_task_seq.ValidationImpl(), true);
  test_mpi_task_seq.PreProcessingImpl();
  test_mpi_task_seq.RunImpl();
  test_mpi_task_seq.PostProcessingImpl();
  // Create data
  double res3 = 0;
  double res4 = 0;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(&left));
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(&right));
    task_data_mpi->inputs_count.emplace_back(2);
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(&res3));
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(&res4));
    task_data_mpi->outputs_count.emplace_back(2);
  }
  // Create Task
  kabalova_v_strongin_mpi::TestMPITaskParallel test_mpi_task_mpi(task_data_mpi, f);
  ASSERT_EQ(test_mpi_task_mpi.ValidationImpl(), true);
  test_mpi_task_mpi.PreProcessingImpl();
  test_mpi_task_mpi.RunImpl();
  test_mpi_task_mpi.PostProcessingImpl();
  if (world.rank() == 0) {
    EXPECT_NEAR(res1, res3, eps);
    EXPECT_NEAR(res2, res4, eps);
  }
}
