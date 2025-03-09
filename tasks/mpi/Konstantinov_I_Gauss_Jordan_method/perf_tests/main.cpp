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
namespace {
std::vector<double> GenerateInvertibleMatrix(int size) {
  std::vector<double> matrix(size * (size + 1));
  std::random_device rd;
  std::mt19937 gen(rd());
  double lowerLimit = -100.0;
  double upperLimit = 100.0;
  std::uniform_real_distribution<> dist(lowerLimit, upperLimit);

  for (int i = 0; i < size; ++i) {
    double row_sum = 0.0;
    double diag = (i * (size + 1) + i);
    for (int j = 0; j < size + 1; ++j) {
      if (i != j) {
        matrix[i * (size + 1) + j] = dist(gen);
        row_sum += std::abs(matrix[i * (size + 1) + j]);
      }
    }
    matrix[diag] = row_sum + 1;
  }

  return matrix;
}
}  // namespace
}  // namespace konstantinov_i_gauss_jordan_method_mpi

TEST(Konstantinov_i_gauss_jordan_method_mpi, test_pipeline_run) {
  boost::mpi::communicator world;
  size_t size = 500;
  std::vector<double> matrix = konstantinov_i_gauss_jordan_method_mpi::GenerateInvertibleMatrix(size);
  std::vector<double> output_data(size, 0.0);
  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(&size));
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    task_data_par->inputs_count.emplace_back(matrix.size() / (size + 1));
    task_data_par->inputs_count.emplace_back(matrix.size());
    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_data.data()));
    task_data_par->outputs_count.emplace_back(output_data.size());
  }

  auto test_mpi =
      std::make_shared<konstantinov_i_gauss_jordan_method_mpi::GaussJordanMethodMpi>(task_data_par);

  ASSERT_EQ(test_mpi->ValidationImpl(), true);
  test_mpi->PreProcessingImpl();
  test_mpi->RunImpl();
  test_mpi->PostProcessingImpl();

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const boost::mpi::timer current_timer;
  perf_attr->current_timer = [&] { return current_timer.elapsed(); };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_mpi);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
    ASSERT_EQ(output_data.size(), size);
  }
}

TEST(Konstantinov_i_gauss_jordan_method_mpi, test_task_run) {
  boost::mpi::communicator world;
  size_t size = 500;
  std::vector<double> matrix = konstantinov_i_gauss_jordan_method_mpi::GenerateInvertibleMatrix(size);
  std::vector<double> output_data(size, 0.0);
  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(&size));
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    task_data_par->inputs_count.emplace_back(matrix.size() / (size + 1));
    task_data_par->inputs_count.emplace_back(matrix.size());
    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_data.data()));
    task_data_par->outputs_count.emplace_back(output_data.size());
  }

  auto test_mpi =
      std::make_shared<konstantinov_i_gauss_jordan_method_mpi::GaussJordanMethodMpi>(task_data_par);

  ASSERT_EQ(test_mpi->ValidationImpl(), true);
  test_mpi->PreProcessingImpl();
  test_mpi->RunImpl();
  test_mpi->PostProcessingImpl();

  // Create Perf attributes
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const boost::mpi::timer current_timer;
  perf_attr->current_timer = [&] { return current_timer.elapsed(); };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_mpi);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
    ASSERT_EQ(output_data.size(), size);
  }
}