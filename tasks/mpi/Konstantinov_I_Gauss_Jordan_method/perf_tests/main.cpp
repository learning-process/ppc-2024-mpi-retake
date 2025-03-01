#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/timer.hpp>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "mpi/Konstantinov_I_Gauss_Jordan_method/include/ops_mpi.hpp"

namespace konstantinov_i_gauss_jordan_method_mpi {

std::vector<double> GetRandomMatrix(int rows, int cols) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_real_distribution<double> dist(-20.0, 20.0);
  std::vector<double> matrix(rows * cols);
  for (int i = 0; i < rows * cols; i++) {
    matrix[i] = dist(gen);
  }
  return matrix;
}

}  // namespace konstantinov_i_gauss_jordan_method_mpi

TEST(konstantinov_i_gauss_jordan_method_mpi, pipeline_run) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  std::vector<double> global_matrix;
  std::vector<double> global_result;

  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();
  int n;

  if (world.rank() == 0) {
    n = 200;
    global_matrix = konstantinov_i_gauss_jordan_method_mpi::GetRandomMatrix(n, n + 1);

    global_result.resize(n * (n + 1));

    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    task_data_par->inputs_count.emplace_back(global_matrix.size());

    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
    task_data_par->inputs_count.emplace_back(1);

    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
    task_data_par->outputs_count.emplace_back(global_result.size());
  }

  auto task_parallel = std::make_shared<konstantinov_i_gauss_jordan_method_mpi::GaussJordanMethodMPI>(task_data_par);
  ASSERT_TRUE(task_parallel->ValidationImpl());
  task_parallel->PreProcessingImpl();
  bool parRunRes = task_parallel->RunImpl();
  task_parallel->PostProcessingImpl();

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const boost::mpi::timer current_timer;
  perf_attr->current_timer = [&] { return current_timer.elapsed(); };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(task_parallel);
  perfAnalyzer->PipelineRun(perf_attr, perf_results);

  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);

    std::vector<double> seq_result(global_result.size(), 0);

    auto task_data_seq = std::make_shared<ppc::core::TaskData>();
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    task_data_seq->inputs_count.emplace_back(global_matrix.size());

    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
    task_data_seq->inputs_count.emplace_back(1);

    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(seq_result.data()));
    task_data_seq->outputs_count.emplace_back(seq_result.size());

    auto task_sequential = 
        std::make_shared<konstantinov_i_gauss_jordan_method_mpi::GaussJordanMethodSeq>(task_data_seq);
    ASSERT_TRUE(task_sequential->ValidationImpl());
    task_sequential->PreProcessingImpl();
    bool seqRunRes = task_sequential->RunImpl();
    task_sequential->PostProcessingImpl();

    if (seqRunRes && parRunRes) {
      ASSERT_EQ(global_result.size(), seq_result.size());
      EXPECT_EQ(global_result, seq_result);
    } else {
      EXPECT_EQ(seqRunRes, parRunRes);
    }
  }
}

TEST(konstantinov_i_gauss_jordan_method_mpi, task_run) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  std::vector<double> global_matrix;
  std::vector<double> global_result;

  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();
  int n;

  if (world.rank() == 0) {
    n = 200;
    global_matrix = konstantinov_i_gauss_jordan_method_mpi::GetRandomMatrix(n, n + 1);

    global_result.resize(n * (n + 1));

    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    task_data_par->inputs_count.emplace_back(global_matrix.size());

    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
    task_data_par->inputs_count.emplace_back(1);

    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
    task_data_par->outputs_count.emplace_back(global_result.size());
  }

  auto task_parallel = std::make_shared<konstantinov_i_gauss_jordan_method_mpi::GaussJordanMethodMPI>(task_data_par);
  ASSERT_TRUE(task_parallel->ValidationImpl());
  task_parallel->PreProcessingImpl();
  bool parRunRes = task_parallel->RunImpl();
  task_parallel->PostProcessingImpl();

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const boost::mpi::timer current_timer;
  perf_attr->current_timer = [&] { return current_timer.elapsed(); };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(task_parallel);
  perf_analyzer->TaskRun(perf_attr, perf_results);

  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);

    std::vector<double> seq_result(global_result.size(), 0);

    auto task_data_seq = std::make_shared<ppc::core::TaskData>();
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    task_data_seq->inputs_count.emplace_back(global_matrix.size());

    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
    task_data_seq->inputs_count.emplace_back(1);

    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(seq_result.data()));
    task_data_seq->outputs_count.emplace_back(seq_result.size());

    auto task_sequential = 
        std::make_shared<konstantinov_i_gauss_jordan_method_mpi::GaussJordanMethodSeq>(task_data_seq);
    ASSERT_TRUE(task_sequential->ValidationImpl());
    task_sequential->PreProcessingImpl();
    bool seqRunRes = task_sequential->RunImpl();
    task_sequential->PostProcessingImpl();

    if (seqRunRes && parRunRes) {
      ASSERT_EQ(global_result.size(), seq_result.size());
      EXPECT_EQ(global_result, seq_result);
    } else {
      EXPECT_EQ(seqRunRes, parRunRes);
    }
  }
}