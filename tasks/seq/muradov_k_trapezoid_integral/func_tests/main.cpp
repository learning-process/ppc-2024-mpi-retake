#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <cstdint>

#include "core/task/include/task.hpp"
#include "seq/muradov_k_trapezoid_integral/include/ops_seq.hpp"

TEST(muradov_k_trap_integral_seq, Test_x2_0_2) {
  std::vector<double> input{0.0, 2.0};
  int n = 1e6;
  double result = 0.0;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  task_data->inputs_count.emplace_back(input.size());
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
  task_data->inputs_count.emplace_back(1);
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result));
  task_data->outputs_count.emplace_back(1);

  muradov_k_trap_integral_seq::TrapezoidalIntegral task(task_data);
  ASSERT_TRUE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();

  EXPECT_NEAR(result, 8.0/3.0, 1e-3);
}

TEST(muradov_k_trap_integral_seq, Invalid_Parameters) {
  std::vector<double> input{5.0, 1.0};
  int n = -100;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  task_data->inputs_count.emplace_back(input.size());
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));

  muradov_k_trap_integral_seq::TrapezoidalIntegral task(task_data);
  ASSERT_FALSE(task.Validation());
}