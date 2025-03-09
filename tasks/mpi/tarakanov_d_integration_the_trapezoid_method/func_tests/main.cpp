// Copyright 2025 Tarakanov Denis
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <memory>

#include "core/task/include/task.hpp"
#include "mpi/tarakanov_d_integration_the_trapezoid_method/include/ops_mpi.hpp"

namespace {
auto CreateTaskData(double* a, double* b, double* h, double* res) {
  auto data = std::make_shared<ppc::core::TaskData>();

  data->inputs.push_back(reinterpret_cast<uint8_t*>(a));
  data->inputs.push_back(reinterpret_cast<uint8_t*>(b));
  data->inputs.push_back(reinterpret_cast<uint8_t*>(h));
  data->inputs_count.push_back(3);

  data->outputs.push_back(reinterpret_cast<uint8_t*>(res));
  data->outputs_count.push_back(1);

  return data;
}
}  // namespace

using namespace tarakanov_d_integration_the_trapezoid_method_mpi;

TEST(tarakanov_d_trapezoid_method_mpi, ValidationPositiveCheck) {
  double a = 0.0;
  double b = 1.0;
  double h = 0.1;
  double res = 0.0;
  auto data = CreateTaskData(&a, &b, &h, &res);

  IntegrationTheTrapezoidMethodMPI task(data);

  EXPECT_TRUE(task.ValidationImpl());
}

TEST(tarakanov_d_trapezoid_method_mpi, ValidationStepNegativeCheck) {
  double a = 0.0;
  double b = 1.0;
  double h = 0.0;
  double res = 0.0;
  auto data = CreateTaskData(&a, &b, &h, &res);

  IntegrationTheTrapezoidMethodMPI task(data);

  EXPECT_TRUE(task.ValidationImpl());
}

TEST(tarakanov_d_trapezoid_method_mpi, PreProcessingPositiveCheck) {
  double a = 0.0;
  double b = 1.0;
  double h = 0.1;
  double res = 0.0;
  auto data = CreateTaskData(&a, &b, &h, &res);

  IntegrationTheTrapezoidMethodMPI task(data);

  EXPECT_TRUE(task.ValidationImpl());
  EXPECT_TRUE(task.PreProcessingImpl());
}

TEST(tarakanov_d_trapezoid_method_mpi, PostProcessingCheck) {
  double a = 0.0;
  double b = 1.0;
  double h = 0.1;
  double res = 0.0;
  auto data = CreateTaskData(&a, &b, &h, &res);

  IntegrationTheTrapezoidMethodMPI task(data);

  EXPECT_TRUE(task.ValidationImpl());
  EXPECT_TRUE(task.PreProcessingImpl());
  EXPECT_TRUE(task.RunImpl());
  EXPECT_TRUE(task.PostProcessingImpl());

  boost::mpi::communicator world;
  if (world.rank() == 0) {
    double output = *reinterpret_cast<double*>(data->outputs[0]);
    bool flag = output == 0.0;
    EXPECT_FALSE(flag);
  }
}

TEST(tarakanov_d_trapezoid_method_mpi, RandomDataCheck) {
  std::srand(static_cast<unsigned int>(std::time(nullptr)));
  auto a = static_cast<double>((std::rand() % 10000) + 1);
  auto b = a + static_cast<double>((std::rand() % 100) + 1);
  auto h = static_cast<double>(std::rand() % 10) / 10;
  double res = 0.0;
  auto data = CreateTaskData(&a, &b, &h, &res);

  IntegrationTheTrapezoidMethodMPI task(data);

  EXPECT_TRUE(task.ValidationImpl());
  EXPECT_TRUE(task.PreProcessingImpl());
  EXPECT_TRUE(task.RunImpl());
  EXPECT_TRUE(task.PostProcessingImpl());
}
