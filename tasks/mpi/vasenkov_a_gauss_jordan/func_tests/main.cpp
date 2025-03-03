#include <gtest/gtest.h>

#include <vector>

#include "core/task/include/task.hpp"
#include "core/util/include/util.hpp"
#include "mpi/vasenkov_a_gauss_jordan/include/ops_mpi.hpp"

TEST(vasenkov_a_gauss_jordan_mpi, three_simple_matrix) {
  boost::mpi::communicator world;

  std::vector<double> global_matrix;
  int n;
  std::vector<double> global_result;

  std::shared_ptr<ppc::core::TaskData> taskDataPar =
      std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_matrix = {1, 2, 1, 10, 4, 8, 3, 20, 2, 5, 9, 30};
    n = 3;

    global_result.resize(n * (n + 1));

    taskDataPar->inputs.emplace_back(
        reinterpret_cast<uint8_t *>(global_matrix.data()));
    taskDataPar->inputs_count.emplace_back(global_matrix.size());

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
    taskDataPar->inputs_count.emplace_back(1);

    taskDataPar->outputs.emplace_back(
        reinterpret_cast<uint8_t *>(global_result.data()));
    taskDataPar->outputs_count.emplace_back(global_result.size());
  }

  auto taskParallel = std::make_shared<
      vasenkov_a_gauss_jordan_mpi::GaussJordanMethodParallelMPI>(taskDataPar);
  ASSERT_TRUE(taskParallel->ValidationImpl());
  taskParallel->PreProcessingImpl();
  bool parRunRes = taskParallel->RunImpl();
  taskParallel->PostProcessingImpl();

  if (world.rank() == 0) {
    std::vector<double> seq_result(global_result.size(), 0);

    auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(
        reinterpret_cast<uint8_t *>(global_matrix.data()));
    taskDataSeq->inputs_count.emplace_back(global_matrix.size());

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
    taskDataSeq->inputs_count.emplace_back(1);

    taskDataSeq->outputs.emplace_back(
        reinterpret_cast<uint8_t *>(seq_result.data()));
    taskDataSeq->outputs_count.emplace_back(seq_result.size());

    auto taskSequential = std::make_shared<
        vasenkov_a_gauss_jordan_mpi::GaussJordanMethodSequentialMPI>(
        taskDataSeq);
    ASSERT_TRUE(taskSequential->ValidationImpl());
    taskSequential->PreProcessingImpl();
    bool seqRunRes = taskSequential->RunImpl();
    taskSequential->PostProcessingImpl();

    if (seqRunRes && parRunRes) {
      ASSERT_EQ(global_result.size(), seq_result.size());
      EXPECT_EQ(global_result, seq_result);
    } else {
      EXPECT_EQ(seqRunRes, parRunRes);
    }
  }
}
