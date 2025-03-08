// Copyright 2025 Tarakanov Denis
#include <gtest/gtest.h>

#include <cstdint>
#include <memory>

#include "core/task/include/task.hpp"
#include "seq/tarakanov_d_integration_the_trapezoid_method/include/ops_seq.hpp"

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

using namespace tarakanov_d_integration_the_trapezoid_method_seq;

TEST(tarakanov_d_trapezoid_method_seq, ValidationPositiveCheck) {
  double a = 0.0;
  double b = 1.0;
  double h = 0.1;
  double res = 0.0;
  auto data = CreateTaskData(&a, &b, &h, &res);

  IntegrationTheTrapezoidMethodSequential task(data);

  EXPECT_TRUE(task.ValidationImpl());
}

TEST(tarakanov_d_trapezoid_method_seq, ValidationPositiveExtraCheck) {
  double a = 0.0;
  double b = 1.0;
  double h = 0.3;
  double res = 0.0;
  auto data = CreateTaskData(&a, &b, &h, &res);

  IntegrationTheTrapezoidMethodSequential task(data);

  EXPECT_TRUE(task.ValidationImpl());
}

TEST(tarakanov_d_trapezoid_method_seq, ValidationStepNegativeCheck) {
  double a = 0.0;
  double b = 1.0;
  double h = 0.0;
  double res = 0.0;
  auto data = CreateTaskData(&a, &b, &h, &res);

  IntegrationTheTrapezoidMethodSequential task(data);

  EXPECT_TRUE(task.ValidationImpl());
}

TEST(tarakanov_d_trapezoid_method_seq, PreProcessingPositiveCheck) {
  double a = 0.0;
  double b = 1.0;
  double h = 0.1;
  double res = 0.0;
  auto data = CreateTaskData(&a, &b, &h, &res);

  IntegrationTheTrapezoidMethodSequential task(data);

  EXPECT_TRUE(task.ValidationImpl());
  EXPECT_TRUE(task.PreProcessingImpl());
}

TEST(tarakanov_d_trapezoid_method_seq, PostProcessingCheck) {
  double a = 0.0;
  double b = 1.0;
  double h = 0.1;
  double res = 0.0;
  auto data = CreateTaskData(&a, &b, &h, &res);

  IntegrationTheTrapezoidMethodSequential task(data);

  EXPECT_TRUE(task.ValidationImpl());
  EXPECT_TRUE(task.PreProcessingImpl());
  EXPECT_TRUE(task.RunImpl());
  EXPECT_TRUE(task.PostProcessingImpl());

  double output = *reinterpret_cast<double*>(data->outputs[0]);
  bool flag = output == 0.0;
  EXPECT_FALSE(flag);
}
