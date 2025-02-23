#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/shishkarev_a_gaussian_method_horizontal_strip_pattern/include/ops_mpi.hpp"

namespace shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi {

std::vector<double> GetRandomMatrix(int sz) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_real_distribution<double> dis(-1000, 1000);
  std::vector<double> matrix_(sz);
  for (int i = 0; i < sz; ++i) {
    matrix_[i] = dis(gen);
  }
  return matrix_;
}

double AxB(int n, int m, std::vector<double> a, std::vector<double> res_) {
  std::vector<double> tmp(m, 0);

  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n - 1; ++j) {
      tmp[i] += a[i * n + j] * res[j];
    }
    tmp[i] -= a[i * n + m];
  }

  double tmp_norm = 0;
  for (int i = 0; i < m; i++) {
    tmp_norm += tmp[i] * tmp[i];
  }
  return sqrt(tmp_norm);
}

}  // namespace shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi

TEST(shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi, test_pipeline_run) {
  boost::mpi::communicator world_;

  const int cols_ = 101;
  const int rows_ = 100;
  std::vector<double> global_matrix(cols_ * rows_);
  std::vector<double> global_res(cols_ - 1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world_.rank() == 0) {
    global_matrix = shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi::GetRandomMatrix(cols_ * rows_);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataPar->inputs_count.emplace_back(global_matrix.size());
    taskDataPar->inputs_count.emplace_back(cols_);
    taskDataPar->inputs_count.emplace_back(rows_);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
  }

  auto MPIGaussHorizontalParallel =
      std::make_shared<shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi::MPIGaussHorizontalParallel>(
          taskDataPar);
  ASSERT_EQ(MPIGaussHorizontalParallel->ValidationImpl(), true);
  MPIGaussHorizontalParallel->PreProcessingImpl();
  MPIGaussHorizontalParallel->RunImpl();
  MPIGaussHorizontalParallel->PostProcessingImpl();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(MPIGaussHorizontalParallel);
  perfAnalyzer->PipelineRun(perfAttr, perfResults);
  if (world_.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perfResults);
    ASSERT_NEAR(shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi::AxB(cols_, rows_, global_matrix, global_res),
                0, 1e-6);
  }
}

TEST(shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi, test_task_run) {
  boost::mpi::communicator world_;

  const int cols_ = 101;
  const int rows_ = 100;
  std::vector<double> global_matrix(cols_ * rows_);
  std::vector<double> global_res(cols_ - 1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world_.rank() == 0) {
    global_matrix = shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi::GetRandomMatrix(cols_ * rows_);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataPar->inputs_count.emplace_back(global_matrix.size());
    taskDataPar->inputs_count.emplace_back(cols_);
    taskDataPar->inputs_count.emplace_back(rows_);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
  }

  auto MPIGaussHorizontalParallel =
      std::make_shared<shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi::MPIGaussHorizontalParallel>(
          taskDataPar);
  ASSERT_EQ(MPIGaussHorizontalParallel->ValidationImpl(), true);
  MPIGaussHorizontalParallel->PreProcessingImpl();
  MPIGaussHorizontalParallel->RunImpl();
  MPIGaussHorizontalParallel->PostProcessingImpl();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(MPIGaussHorizontalParallel);
  perfAnalyzer->TaskRun(perfAttr, perfResults);
  if (world_.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perfResults);
    ASSERT_NEAR(shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi::AxB(cols_, rows_, global_matrix, global_res),
                0, 1e-6);
  }
}