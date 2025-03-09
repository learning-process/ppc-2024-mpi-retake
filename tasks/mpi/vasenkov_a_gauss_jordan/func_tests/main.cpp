#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "mpi/vasenkov_a_gauss_jordan/include/ops_mpi.hpp"


TEST(vasenkov_a_gauss_jordan_mpi, three_simple_matrix) {
  boost::mpi::communicator world;

  std::vector<double> global_matrix;
  int n = 0;
  std::vector<double> global_result;

  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_matrix = {1, 2, 1, 10, 4, 8, 3, 20, 2, 5, 9, 30};
    n = 3;

    global_result.resize(n * (n + 1));

    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_matrix.data()));
    task_data_par->inputs_count.emplace_back(global_matrix.size());

    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
    task_data_par->inputs_count.emplace_back(1);

    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_result.data()));
    task_data_par->outputs_count.emplace_back(global_result.size());
  }

  auto task_parallel = std::make_shared<vasenkov_a_gauss_jordan_mpi::GaussJordanMethodParallelMPI>(task_data_par);
  ASSERT_TRUE(task_parallel->ValidationImpl());
  task_parallel->PreProcessingImpl();
  bool par_run_res = task_parallel->RunImpl();
  task_parallel->PostProcessingImpl();

  if (world.rank() == 0) {
    std::vector<double> seq_result(global_result.size(), 0);

    auto task_data_seq = std::make_shared<ppc::core::TaskData>();
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_matrix.data()));
    task_data_seq->inputs_count.emplace_back(global_matrix.size());

    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
    task_data_seq->inputs_count.emplace_back(1);

    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(seq_result.data()));
    task_data_seq->outputs_count.emplace_back(seq_result.size());

    auto task_sequential = std::make_shared<vasenkov_a_gauss_jordan_mpi::GaussJordanMethodSequentialMPI>(task_data_seq);
    ASSERT_TRUE(task_sequential->ValidationImpl());
    task_sequential->PreProcessingImpl();
    bool seq_run_res = task_sequential->RunImpl();
    task_sequential->PostProcessingImpl();

    if (seq_run_res && par_run_res) {
      ASSERT_EQ(global_result.size(), seq_result.size());
      EXPECT_EQ(global_result, seq_result);
    } else {
      EXPECT_EQ(seq_run_res, par_run_res);
    }
  }
}
