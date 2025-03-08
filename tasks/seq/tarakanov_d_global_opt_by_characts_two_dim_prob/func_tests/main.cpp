// @copyright Tarakanov Denis
#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/tarakanov_d_global_opt_by_characts_two_dim_prob/include/ops_seq.hpp"

namespace {
std::vector<double> CreateFunc(int min, int max) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dist(min, max);
  return {dist(gen), dist(gen)};
}

std::vector<double> CreateConstr(int min, int max, int count) {
  std::vector<double> constr(3 * count);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dist(min, max);

  for (double& val : constr) {
    val = dist(gen);
  }
  return constr;
}

void RunTest(const std::vector<double>& area, int constraint_count, int mode) {
  double step = 0.3;
  auto func = CreateFunc(-10, 10);
  auto constraint = CreateConstr(-10, 10, constraint_count);
  std::vector<double> out_s(1, 0);

  auto create_task_data = [&]() {
    auto task_data = std::make_shared<ppc::core::TaskData>();
    task_data->inputs = {reinterpret_cast<uint8_t*>(const_cast<double*>(area.data())),
                         reinterpret_cast<uint8_t*>(func.data()), reinterpret_cast<uint8_t*>(constraint.data()),
                         reinterpret_cast<uint8_t*>(&step)};
    task_data->inputs_count = {static_cast<unsigned int>(constraint_count), static_cast<unsigned int>(mode)};
    task_data->outputs = {reinterpret_cast<uint8_t*>(out_s.data())};
    task_data->outputs_count = {static_cast<unsigned int>(out_s.size())};
    return task_data;
  };

  auto task_data_seq = create_task_data();
  tarakanov_d_global_opt_two_dim_prob_seq::GlobalOptSequential test_class_seq(task_data_seq);

  ASSERT_TRUE(test_class_seq.Validation());
  ASSERT_TRUE(test_class_seq.PreProcessing());
  ASSERT_TRUE(test_class_seq.Run());
  ASSERT_TRUE(test_class_seq.PostProcessing());
}
}  // namespace

TEST(tarakanov_d_global_opt_two_dim_prob_seq, test_min_0) { RunTest({1e-7, 2e-7, 1e-7, 2e-7}, 1, 0); }
TEST(tarakanov_d_global_opt_two_dim_prob_seq, test_min_1) { RunTest({-10, 10, -10, 10}, 36, 0); }
TEST(tarakanov_d_global_opt_two_dim_prob_seq, test_min_2) { RunTest({-17, 6, 13, 23}, 24, 0); }
TEST(tarakanov_d_global_opt_two_dim_prob_seq, test_max_1) { RunTest({-20, -10, -20, -10}, 1, 1); }
TEST(tarakanov_d_global_opt_two_dim_prob_seq, test_min_3) { RunTest({30, 40, 30, 40}, 36, 0); }
