#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <random>
#include <vector>

#include "mpi/tarakanov_d_global_opt_by_characts_two_dim_prob/include/ops_mpi.hpp"

static std::vector<double> createFunc(int min, int max) {
    std::vector<double> func(2, 0);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist(min, max);

    for (int i = 0; i < 2; i++) {
        func[i] = dist(gen);
    }
    return func;
}

static std::vector<double> createConstr(int min, int max, int count) {
    std::vector<double> constr(3 * count, 0);
    srand(time(nullptr));
    for (int i = 0; i < 3 * count; i++) {
        constr[i] = (min + rand() % (max - min + 1));
    }
    return constr;
}

struct TestParams {
    std::vector<double> area;
    std::vector<double> func;
    std::vector<double> constraint;
    double step;
    int mode;
    int constraint_count;
};

namespace {
std::shared_ptr<ppc::core::TaskData> createTaskData(const TestParams& params, std::vector<double>& out) {
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<double*>(params.area.data())));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<double*>(params.func.data())));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<double*>(params.constraint.data())));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<double*>(&params.step)));
  
  taskData->inputs_count.emplace_back(params.constraint_count);
  taskData->inputs_count.emplace_back(params.mode);
  
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskData->outputs_count.emplace_back(out.size());
  
  return taskData;
}
}

class OptimParMPITest : public testing::TestWithParam<TestParams> {};

TEST_P(OptimParMPITest, TestOptimization) {
    TestParams params = GetParam();
    boost::mpi::communicator world;
    std::vector<double> out = {0};
    std::vector<double> out_s = {0};

    std::shared_ptr<ppc::core::TaskData> taskDataPar;
    if (world.rank() == 0) {
        taskDataPar = createTaskData(params, out);
    } else {
        taskDataPar = std::make_shared<ppc::core::TaskData>();
    }

    tarakanov_d_global_opt_two_dim_prob_mpi::GlobalOptMpi testClassPar(taskDataPar);
    ASSERT_EQ(testClassPar.ValidationImpl(), true);
    testClassPar.PreProcessingImpl();
    testClassPar.RunImpl();
    testClassPar.PostProcessingImpl();

    if (world.rank() == 0) {
        std::shared_ptr<ppc::core::TaskData> taskDataSeq = createTaskData(params, out_s);
        tarakanov_d_global_opt_two_dim_prob_mpi::GlobalOptSequential testClassSeq(taskDataSeq);
        ASSERT_EQ(testClassSeq.ValidationImpl(), true);
        testClassSeq.PreProcessingImpl();
        testClassSeq.RunImpl();
        testClassSeq.PostProcessingImpl();
        ASSERT_EQ(out_s, out);
    }
}

INSTANTIATE_TEST_SUITE_P(
    tarakanov_d_global_opt_two_dim_prob_mpi,
    OptimParMPITest,
    testing::Values(
        TestParams{{-10, 10, -10, 10}, createFunc(-10, 10), createConstr(-10, 10, 36), 0.3, 0, 36},
        TestParams{{-17, 6, 13, 23}, createFunc(-10, 10), createConstr(-10, 10, 24), 0.3, 0, 24},
        TestParams{{-20, -10, -20, -10}, createFunc(-10, 10), createConstr(1, 3, 1), 0.3, 1, 1},
        TestParams{{30, 40, 30, 40}, createFunc(-10, 10), createConstr(-10, -1, 36), 0.3, 0, 36},
        TestParams{{0.0000001, 0.0000002, 0.0000001, 0.0000002}, createFunc(-10, 10), createConstr(-10, 10, 1), 0.3, 0, 1}
    )
);