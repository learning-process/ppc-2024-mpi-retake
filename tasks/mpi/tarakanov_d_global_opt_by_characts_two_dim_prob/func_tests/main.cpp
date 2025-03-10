#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "mpi/tarakanov_d_global_opt_by_characts_two_dim_prob/include/ops_mpi.hpp"

struct TestParams {
  std::vector<double> area;
  std::vector<double> func;
  std::vector<double> constraint;
  double step;
  int mode;
  int constraint_count;
};

namespace {
std::vector<double> CreateFunc(int min, int max) {
  std::vector<double> func(2, 0);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dist(min, max);

  for (int i = 0; i < 2; i++) {
    func[i] = dist(gen);
  }
  return func;
}

std::vector<double> CreateConstr(int min, int max, int count) {
  std::vector<double> constr(3 * count, 0);
  srand(time(nullptr));
  for (int i = 0; i < 3 * count; i++) {
    constr[i] = (min + rand() % (max - min + 1));
  }
  return constr;
}

std::shared_ptr<ppc::core::TaskData> CreateTaskData(const TestParams& params, std::vector<double>& out) {
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();

  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<double*>(params.area.data())));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<double*>(params.func.data())));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<double*>(params.constraint.data())));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<double*>(&params.step)));

  task_data->inputs_count.emplace_back(params.constraint_count);
  task_data->inputs_count.emplace_back(params.mode);

  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  return task_data;
}
}  // namespace

TEST(OptimMPITest, TestOptimization_Case1) {
  TestParams params{.area = {-10, 10, -10, 10},
                    .func = CreateFunc(-10, 10),
                    .constraint = CreateConstr(-10, 10, 36),
                    .step = 0.3,
                    .mode = 0,
                    .constraint_count = 36};
  boost::mpi::communicator world;
  std::vector<double> out = {0};
  std::vector<double> out_s = {0};

  std::shared_ptr<ppc::core::TaskData> task_data_par;
  if (world.rank() == 0) {
    task_data_par = CreateTaskData(params, out);
  } else {
    task_data_par = std::make_shared<ppc::core::TaskData>();
  }

  tarakanov_d_global_opt_two_dim_prob_mpi::GlobalOptMpi test_class_par(task_data_par);
  ASSERT_EQ(test_class_par.ValidationImpl(), true);
  test_class_par.PreProcessingImpl();
  test_class_par.RunImpl();
  test_class_par.PostProcessingImpl();

  if (world.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> task_data_seq = CreateTaskData(params, out_s);
    tarakanov_d_global_opt_two_dim_prob_mpi::GlobalOptSequential test_class_seq(task_data_seq);
    ASSERT_EQ(test_class_seq.ValidationImpl(), true);
    test_class_seq.PreProcessingImpl();
    test_class_seq.RunImpl();
    test_class_seq.PostProcessingImpl();
    ASSERT_EQ(out_s, out);
  }
}

TEST(OptimMPITest, TestOptimization_Case2) {
  TestParams params{.area = {-17, 6, 13, 23},
                    .func = CreateFunc(-10, 10),
                    .constraint = CreateConstr(-10, 10, 24),
                    .step = 0.3,
                    .mode = 0,
                    .constraint_count = 24};
  boost::mpi::communicator world;
  std::vector<double> out = {0};
  std::vector<double> out_s = {0};

  std::shared_ptr<ppc::core::TaskData> task_data_par;
  if (world.rank() == 0) {
    task_data_par = CreateTaskData(params, out);
  } else {
    task_data_par = std::make_shared<ppc::core::TaskData>();
  }

  tarakanov_d_global_opt_two_dim_prob_mpi::GlobalOptMpi test_class_par(task_data_par);
  ASSERT_EQ(test_class_par.ValidationImpl(), true);
  test_class_par.PreProcessingImpl();
  test_class_par.RunImpl();
  test_class_par.PostProcessingImpl();

  if (world.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> task_data_seq = CreateTaskData(params, out_s);
    tarakanov_d_global_opt_two_dim_prob_mpi::GlobalOptSequential test_class_seq(task_data_seq);
    ASSERT_EQ(test_class_seq.ValidationImpl(), true);
    test_class_seq.PreProcessingImpl();
    test_class_seq.RunImpl();
    test_class_seq.PostProcessingImpl();
    ASSERT_EQ(out_s, out);
  }
}

TEST(OptimMPITest, TestOptimization_Case3) {
  TestParams params{.area = {-20, -10, -20, -10},
                    .func = CreateFunc(-10, 10),
                    .constraint = CreateConstr(1, 3, 1),
                    .step = 0.3,
                    .mode = 1,
                    .constraint_count = 1};
  boost::mpi::communicator world;
  std::vector<double> out = {0};
  std::vector<double> out_s = {0};

  std::shared_ptr<ppc::core::TaskData> task_data_par;
  if (world.rank() == 0) {
    task_data_par = CreateTaskData(params, out);
  } else {
    task_data_par = std::make_shared<ppc::core::TaskData>();
  }

  tarakanov_d_global_opt_two_dim_prob_mpi::GlobalOptMpi test_class_par(task_data_par);
  ASSERT_EQ(test_class_par.ValidationImpl(), true);
  test_class_par.PreProcessingImpl();
  test_class_par.RunImpl();
  test_class_par.PostProcessingImpl();

  if (world.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> task_data_seq = CreateTaskData(params, out_s);
    tarakanov_d_global_opt_two_dim_prob_mpi::GlobalOptSequential test_class_seq(task_data_seq);
    ASSERT_EQ(test_class_seq.ValidationImpl(), true);
    test_class_seq.PreProcessingImpl();
    test_class_seq.RunImpl();
    test_class_seq.PostProcessingImpl();
    ASSERT_EQ(out_s, out);
  }
}

TEST(OptimMPITest, TestOptimization_Case4) {
  TestParams params{.area = {30, 40, 30, 40},
                    .func = CreateFunc(-10, 10),
                    .constraint = CreateConstr(-10, -1, 36),
                    .step = 0.3,
                    .mode = 0,
                    .constraint_count = 36};
  boost::mpi::communicator world;
  std::vector<double> out = {0};
  std::vector<double> out_s = {0};

  std::shared_ptr<ppc::core::TaskData> task_data_par;
  if (world.rank() == 0) {
    task_data_par = CreateTaskData(params, out);
  } else {
    task_data_par = std::make_shared<ppc::core::TaskData>();
  }

  tarakanov_d_global_opt_two_dim_prob_mpi::GlobalOptMpi test_class_par(task_data_par);
  ASSERT_EQ(test_class_par.ValidationImpl(), true);
  test_class_par.PreProcessingImpl();
  test_class_par.RunImpl();
  test_class_par.PostProcessingImpl();

  if (world.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> task_data_seq = CreateTaskData(params, out_s);
    tarakanov_d_global_opt_two_dim_prob_mpi::GlobalOptSequential test_class_seq(task_data_seq);
    ASSERT_EQ(test_class_seq.ValidationImpl(), true);
    test_class_seq.PreProcessingImpl();
    test_class_seq.RunImpl();
    test_class_seq.PostProcessingImpl();
    ASSERT_EQ(out_s, out);
  }
}

TEST(OptimMPITest, TestOptimization_Case5) {
  TestParams params{.area = {0.0000001, 0.0000002, 0.0000001, 0.0000002},
                    .func = CreateFunc(-10, 10),
                    .constraint = CreateConstr(-10, 10, 1),
                    .step = 0.3,
                    .mode = 0,
                    .constraint_count = 1};
  boost::mpi::communicator world;
  std::vector<double> out = {0};
  std::vector<double> out_s = {0};

  std::shared_ptr<ppc::core::TaskData> task_data_par;
  if (world.rank() == 0) {
    task_data_par = CreateTaskData(params, out);
  } else {
    task_data_par = std::make_shared<ppc::core::TaskData>();
  }

  tarakanov_d_global_opt_two_dim_prob_mpi::GlobalOptMpi test_class_par(task_data_par);
  ASSERT_EQ(test_class_par.ValidationImpl(), true);
  test_class_par.PreProcessingImpl();
  test_class_par.RunImpl();
  test_class_par.PostProcessingImpl();

  if (world.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> task_data_seq = CreateTaskData(params, out_s);
    tarakanov_d_global_opt_two_dim_prob_mpi::GlobalOptSequential test_class_seq(task_data_seq);
    ASSERT_EQ(test_class_seq.ValidationImpl(), true);
    test_class_seq.PreProcessingImpl();
    test_class_seq.RunImpl();
    test_class_seq.PostProcessingImpl();
    ASSERT_EQ(out_s, out);
  }
}