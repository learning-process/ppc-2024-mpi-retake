#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "mpi/sedova_o_mult_matrices_ccs/include/ops_mpi.hpp"

namespace sedova_o_test_task_mpi {
std::vector<std::vector<double>> GenerateRandomMatrix(int rows, int columns) {
  std::vector<std::vector<double>> result(rows, std::vector<double>(columns, 0));
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < columns; ++j) {
      result[i][j] = (static_cast<double>(rand()) / RAND_MAX) * 2000 - 1000;
    }
  }
  return result;
}
}  // namespace sedova_o_test_task_mpi

TEST(sedova_o_test_task_mpi, test_pipeline_run) {
  boost::mpi::communicator world;
  std::vector<std::vector<double>> matrix_A = sedova_o_test_task_mpi::GenerateRandomMatrix(50, 100);
  std::vector<std::vector<double>> matrix_B = sedova_o_test_task_mpi::GenerateRandomMatrix(100, 50);
  std::vector<double> A;
  std::vector<int> row_ind_A;
  std::vector<int> col_ind_A;
  sedova_o_test_task_mpi::ConvertToCCS(matrix_A, A, row_ind_A, col_ind_A);
  std::vector<double> B;
  std::vector<int> row_ind_B;
  std::vector<int> col_ind_B;
  sedova_o_test_task_mpi::ConvertToCCS(matrix_B, B, row_ind_B, col_ind_B);
  // Create TaskData
  auto task_data_par = std::make_shared<ppc::core::TaskData>();
  std::vector<std::vector<double>> out_par(matrix_A.size(), std::vector<double>(matrix_B[0].size(), 0));
  if (world.rank() == 0) {
    sedova_o_test_task_mpi::FillData(task_data_par, matrix_A.size(), matrix_A[0].size(), matrix_B.size(),
                                     matrix_B[0].size(), A, row_ind_A, col_ind_A, B, row_ind_B, col_ind_B, out_par);
  }
  auto TestMpiTaskParallel = std::make_shared<sedova_o_test_task_mpi::TestTaskMPI>(task_data_par);
  ASSERT_EQ(TestMpiTaskParallel.ValidationImpl(), true);
  TestMpiTaskParallel.PreProcessingImpl();
  TestMpiTaskParallel.RunImpl();
  TestMpiTaskParallel.PostProcessingImpl();
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(TestMpiTaskParallel);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
}

TEST(sedova_o_test_task_mpi, test_task_run) {
  boost::mpi::communicator world;
  std::vector<std::vector<double>> matrix_A = sedova_o_test_task_mpi::GenerateRandomMatrix(50, 100);
  std::vector<std::vector<double>> matrix_B = sedova_o_test_task_mpi::GenerateRandomMatrix(100, 50);
  std::vector<double> A;
  std::vector<int> row_ind_A;
  std::vector<int> col_ind_A;
  sedova_o_test_task_mpi::ConvertToCCS(matrix_A, A, row_ind_A, col_ind_A);
  std::vector<double> B;
  std::vector<int> row_ind_B;
  std::vector<int> col_ind_B;
  sedova_o_test_task_mpi::ConvertToCCS(matrix_B, B, row_ind_B, col_ind_B);
  // Create TaskData
  auto task_data_par = std::make_shared<ppc::core::TaskData>();
  std::vector<std::vector<double>> out_par(matrix_A.size(), std::vector<double>(matrix_B[0].size(), 0));
  if (world.rank() == 0) {
    sedova_o_test_task_mpi::FillData(task_data_par, matrix_A.size(), matrix_A[0].size(), matrix_B.size(),
                                     matrix_B[0].size(), A, row_ind_A, col_ind_A, B, row_ind_B, col_ind_B, out_par);
  }
  auto TestMpiTaskParallel = std::make_shared<sedova_o_test_task_mpi::TestTaskMPI>(task_data_par);
  ASSERT_EQ(TestMpiTaskParallel.ValidationImpl(), true);
  TestMpiTaskParallel.PreProcessingImpl();
  TestMpiTaskParallel.RunImpl();
  TestMpiTaskParallel.PostProcessingImpl();
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };
  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(TestMpiTaskParallel);
  perf_analyzer->TaskRun(perf_attr, perf_results);
}