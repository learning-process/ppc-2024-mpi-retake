#include <gtest/gtest.h>

#include <algorithm>
#include <boost/mpi/communicator.hpp>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "mpi/deryabin_m_cannons_algorithm/include/ops_mpi.hpp"

TEST(deryabin_m_cannons_algorithm_mpi, test_pipeline_run_Mpi) {
  boost::mpi::communicator world;
  constexpr size_t kMatrixSize = 500;
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> distribution(-100, 100);
  std::vector<double> input_matrix_a(kMatrixSize * kMatrixSize);
  std::vector<double> input_matrix_b(kMatrixSize * kMatrixSize);
  std::ranges::generate(input_matrix_a.begin(), input_matrix_a.end(), [&] { return distribution(gen); });
  std::ranges::generate(input_matrix_b.begin(), input_matrix_b.end(), [&] { return distribution(gen); });
  std::vector<double> output_matrix_c(kMatrixSize * kMatrixSize);
  std::vector<std::vector<double>> out_matrix_c(1, output_matrix_c);
  std::vector<double> true_solution(kMatrixSize * kMatrixSize);
  std::vector<std::vector<double>> true_sol(1, true_solution);
  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix_a.data()));
  task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix_b.data()));
  task_data_mpi->inputs_count.emplace_back(input_matrix_a.size());
  task_data_mpi->inputs_count.emplace_back(input_matrix_b.size());
  task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_matrix_c.data()));
  task_data_mpi->outputs_count.emplace_back(out_matrix_c.size());
  if (world.rank() == 0) {
    auto task_data_seq = std::make_shared<ppc::core::TaskData>();
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix_a.data()));
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix_b.data()));
    task_data_seq->inputs_count.emplace_back(input_matrix_a.size());
    task_data_seq->inputs_count.emplace_back(input_matrix_b.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(true_sol.data()));
    task_data_seq->outputs_count.emplace_back(true_sol.size());
    deryabin_m_cannons_algorithm_mpi::CannonsAlgorithmMPITaskSequential test_mpi_task_sequential(task_data_seq);
    ASSERT_EQ(test_mpi_task_sequential.Validation(), true);
    test_mpi_task_sequential.PreProcessing();
    test_mpi_task_sequential.Run();
    test_mpi_task_sequential.PostProcessing();
  }

  auto test_mpi_task_parallel =
      std::make_shared<deryabin_m_cannons_algorithm_mpi::CannonsAlgorithmMPITaskParallel>(task_data_mpi);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_mpi_task_parallel);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
    ASSERT_EQ(true_sol[0], out_matrix_c[0]);
  }
}

TEST(deryabin_m_cannons_algorithm_mpi, test_task_run_Mpi) {
  boost::mpi::communicator world;
  constexpr size_t kMatrixSize = 500;
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> distribution(-100, 100);
  std::vector<double> input_matrix_a(kMatrixSize * kMatrixSize);
  std::vector<double> input_matrix_b(kMatrixSize * kMatrixSize);
  std::ranges::generate(input_matrix_a.begin(), input_matrix_a.end(), [&] { return distribution(gen); });
  std::ranges::generate(input_matrix_b.begin(), input_matrix_b.end(), [&] { return distribution(gen); });
  std::vector<double> output_matrix_c(kMatrixSize * kMatrixSize);
  std::vector<std::vector<double>> out_matrix_c(1, output_matrix_c);
  std::vector<double> true_solution(kMatrixSize * kMatrixSize);
  std::vector<std::vector<double>> true_sol(1, true_solution);
  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix_a.data()));
  task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix_b.data()));
  task_data_mpi->inputs_count.emplace_back(input_matrix_a.size());
  task_data_mpi->inputs_count.emplace_back(input_matrix_b.size());
  task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_matrix_c.data()));
  task_data_mpi->outputs_count.emplace_back(out_matrix_c.size());
  if (world.rank() == 0) {
    auto task_data_seq = std::make_shared<ppc::core::TaskData>();
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix_a.data()));
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix_b.data()));
    task_data_seq->inputs_count.emplace_back(input_matrix_a.size());
    task_data_seq->inputs_count.emplace_back(input_matrix_b.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(true_sol.data()));
    task_data_seq->outputs_count.emplace_back(true_sol.size());
    deryabin_m_cannons_algorithm_mpi::CannonsAlgorithmMPITaskSequential test_mpi_task_sequential(task_data_seq);
    ASSERT_EQ(test_mpi_task_sequential.Validation(), true);
    test_mpi_task_sequential.PreProcessing();
    test_mpi_task_sequential.Run();
    test_mpi_task_sequential.PostProcessing();
  }

  auto test_mpi_task_parallel =
      std::make_shared<deryabin_m_cannons_algorithm_mpi::CannonsAlgorithmMPITaskParallel>(task_data_mpi);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_mpi_task_parallel);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
    ASSERT_EQ(true_sol[0], out_matrix_c[0]);
  }
}
