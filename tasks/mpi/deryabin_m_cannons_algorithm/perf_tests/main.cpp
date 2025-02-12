#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <vector>

#include <boost/mpi/communicator.hpp>
#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "mpi/deryabin_m_cannons_algorithm/include/ops_mpi.hpp"
#include <boost/mpi/timer.hpp>

TEST(deryabin_m_cannons_algorithm_mpi, test_pipeline_run) {
  boost::mpi::communicator world;
  std::vector<double> input_matrix_a = std::vector<double>(10000, 0);
  std::vector<double> input_matrix_b = std::vector<double>(10000, 0);
  std::vector<double> output_matrix_c = std::vector<double>(10000, 0);
  for (unsigned short dimension = 0; dimension < 100; dimension++) {
    input_matrix_a[dimension * 101] = 1;
    input_matrix_b[dimension * 101] = 1;
  }
  std::vector<std::vector<double>> out_matrix_c(1, output_matrix_c);
  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix_a.data()));
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix_b.data()));
    task_data_mpi->inputs_count.emplace_back(input_matrix_a.size());
    task_data_mpi->inputs_count.emplace_back(input_matrix_b.size());
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_matrix_c.data()));
    task_data_mpi->outputs_count.emplace_back(out_matrix_c.size());
  }

  auto test_mpi_task_parallel =
      std::make_shared<deryabin_m_cannons_algorithm_mpi::CannonsAlgorithmMPITaskParallel>(task_data_mpi);
  ASSERT_EQ(test_mpi_task_parallel->Validation(), true);
  test_mpi_task_parallel->PreProcessing();
  test_mpi_task_parallel->Run();
  test_mpi_task_parallel->PostProcessing();

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const boost::mpi::timer current_timer;
  perf_attr->current_timer = [&] { return current_timer.elapsed(); };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_mpi_task_parallel);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
    ASSERT_EQ(input_matrix_a, out_matrix_c[0]);
  }
}

TEST(deryabin_m_cannons_algorithm_mpi, test_task_run) {
  boost::mpi::communicator world;
  std::vector<double> input_matrix_a = std::vector<double>(10000, 0);
  std::vector<double> input_matrix_b = std::vector<double>(10000, 0);
  std::vector<double> output_matrix_c = std::vector<double>(10000, 0);
  for (unsigned short dimension = 0; dimension < 100; dimension++) {
    input_matrix_a[dimension * 101] = 1;
    input_matrix_b[dimension * 101] = 1;
  }
  std::vector<std::vector<double>> out_matrix_c(1, output_matrix_c);
  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix_a.data()));
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix_b.data()));
    task_data_mpi->inputs_count.emplace_back(input_matrix_a.size());
    task_data_mpi->inputs_count.emplace_back(input_matrix_b.size());
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_matrix_c.data()));
    task_data_mpi->outputs_count.emplace_back(out_matrix_c.size());
  }

  auto test_mpi_task_parallel =
      std::make_shared<deryabin_m_cannons_algorithm_mpi::CannonsAlgorithmMPITaskParallel>(task_data_mpi);
  ASSERT_EQ(test_mpi_task_parallel->Validation(), true);
  test_mpi_task_parallel->PreProcessing();
  test_mpi_task_parallel->Run();
  test_mpi_task_parallel->PostProcessing();

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const boost::mpi::timer current_timer;
  perf_attr->current_timer = [&] { return current_timer.elapsed(); };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_mpi_task_parallel);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
    ASSERT_EQ(input_matrix_a, out_matrix_c[0]);
  }
}
