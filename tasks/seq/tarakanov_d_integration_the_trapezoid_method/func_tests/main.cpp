// Copyright 2025 Tarakanov Denis
#include <gtest/gtest.h>

#include <memory>

#include "seq/tarakanov_d_integration_the_trapezoid_method/include/ops_seq.hpp"

namespace tarakanov_d_integration_the_trapezoid_method_seq {
static auto createTaskData(double* a, double* b, double* h, double* res) {
  auto data = std::make_shared<ppc::core::TaskData>();

  data->inputs.push_back(reinterpret_cast<uint8_t*>(a));
  data->inputs.push_back(reinterpret_cast<uint8_t*>(b));
  data->inputs.push_back(reinterpret_cast<uint8_t*>(h));
  data->inputs_count.push_back(3);

  data->outputs.push_back(reinterpret_cast<uint8_t*>(res));
  data->outputs_count.push_back(1);

  return data;
}
}  // namespace tarakanov_d_integration_the_trapezoid_method_seq

using namespace tarakanov_d_integration_the_trapezoid_method_seq;

TEST(tarakanov_d_integration_the_trapezoid_method_func_test, ValidationPositiveCheck) {
  double a = 0.0;
  double b = 1.0;
  double h = 0.1;
  double res = 0.0;
  auto data = createTaskData(&a, &b, &h, &res);

  IntegrationTheTrapezoidMethodSequential task(data);

  EXPECT_TRUE(task.ValidationImpl());
}

TEST(tarakanov_d_integration_the_trapezoid_method_func_test, ValidationStepNegativeCheck) {
  double a = 0.0;
  double b = 1.0;
  double h = 0.0;
  double res = 0.0;
  auto data = createTaskData(&a, &b, &h, &res);

  IntegrationTheTrapezoidMethodSequential task(data);

  EXPECT_TRUE(task.ValidationImpl());
}

TEST(tarakanov_d_integration_the_trapezoid_method_func_test, PreProcessingPositiveCheck) {
  double a = 0.0;
  double b = 1.0;
  double h = 0.1;
  double res = 0.0;
  auto data = createTaskData(&a, &b, &h, &res);

  IntegrationTheTrapezoidMethodSequential task(data);

  EXPECT_TRUE(task.ValidationImpl());
  EXPECT_TRUE(task.PreProcessingImpl());
}

TEST(tarakanov_d_integration_the_trapezoid_method_func_test, PostProcessingCheck) {
  double a = 0.0;
  double b = 1.0;
  double h = 0.1;
  double res = 0.0;
  auto data = createTaskData(&a, &b, &h, &res);

  IntegrationTheTrapezoidMethodSequential task(data);

  EXPECT_TRUE(task.ValidationImpl());
  EXPECT_TRUE(task.PreProcessingImpl());
  EXPECT_TRUE(task.RunImpl());
  EXPECT_TRUE(task.PostProcessingImpl());

  double output = *reinterpret_cast<double*>(data->outputs[0]);
  bool flag = output == 0.0;
  EXPECT_FALSE(flag);
}
