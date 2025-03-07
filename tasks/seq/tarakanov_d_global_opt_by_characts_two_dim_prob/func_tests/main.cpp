// @copyright Tarakanov Denis
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <random>
#include <vector>

#include "seq/tarakanov_d_global_opt_by_characts_two_dim_prob/include/ops_seq.hpp"

namespace {
std::vector<double> createFunc(int min, int max) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dist(min, max);
  return {dist(gen), dist(gen)};
}

std::vector<double> createConstr(int min, int max, int count) {
  std::vector<double> constr(3 * count);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dist(min, max);

  for (double& val : constr) {
    val = dist(gen);
  }
  return constr;
}

void runTest(const std::vector<double>& area, int constraintCount, int mode) {
  double step = 0.3;
  auto func = createFunc(-10, 10);
  auto constraint = createConstr(-10, 10, constraintCount);
  std::vector<double> out_s(1, 0);

  auto createTaskData = [&]() {
    auto taskData = std::make_shared<ppc::core::TaskData>();
    taskData->inputs = {reinterpret_cast<uint8_t*>(const_cast<double*>(area.data())),
                        reinterpret_cast<uint8_t*>(func.data()), reinterpret_cast<uint8_t*>(constraint.data()),
                        reinterpret_cast<uint8_t*>(&step)};
    taskData->inputs_count = {static_cast<unsigned int>(constraintCount), static_cast<unsigned int>(mode)};
    taskData->outputs = {reinterpret_cast<uint8_t*>(out_s.data())};
    taskData->outputs_count = {static_cast<unsigned int>(out_s.size())};
    return taskData;
  };

  auto taskDataSeq = createTaskData();
  tarakanov_d_global_opt_two_dim_prob_seq::GlobalOptSequential testClassSeq(taskDataSeq);

  ASSERT_TRUE(testClassSeq.Validation());
  ASSERT_TRUE(testClassSeq.PreProcessing());
  ASSERT_TRUE(testClassSeq.Run());
  ASSERT_TRUE(testClassSeq.PostProcessing());
}
}  // namespace

TEST(tarakanov_d_global_opt_two_dim_prob_seq, test_min_0) { runTest({1e-7, 2e-7, 1e-7, 2e-7}, 1, 0); }
TEST(tarakanov_d_global_opt_two_dim_prob_seq, test_min_1) { runTest({-10, 10, -10, 10}, 36, 0); }
TEST(tarakanov_d_global_opt_two_dim_prob_seq, test_min_2) { runTest({-17, 6, 13, 23}, 24, 0); }
TEST(tarakanov_d_global_opt_two_dim_prob_seq, test_max_1) { runTest({-20, -10, -20, -10}, 1, 1); }
TEST(tarakanov_d_global_opt_two_dim_prob_seq, test_min_3) { runTest({30, 40, 30, 40}, 36, 0); }
