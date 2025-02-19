#include <gtest/gtest.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "mpi/deryabin_m_cannons_algorithm/include/ops_mpi.hpp"

TEST(deryabin_m_cannons_algorithm_mpi, test_pipeline_run_Mpi) {
  constexpr size_t kMatrixSize = 256;
  std::vector<double> input_matrix_a = std::vector<double>(kMatrixSize * kMatrixSize, 0);
  std::vector<double> input_matrix_b = std::vector<double>(kMatrixSize * kMatrixSize, 0);
  std::vector<double> output_matrix_c = std::vector<double>(kMatrixSize * kMatrixSize, 0);
  std::vector<std::vector<double>> out_matrix_c(1, output_matrix_c);
  std::vector<double> true_solution = std::vector<double>(kMatrixSize * kMatrixSize, 0);
  for (unsigned short i = 0; i < kMatrixSize; i++) {
    for (unsigned short j = 0; j < kMatrixSize; j++) {
      input_matrix_a[j + (i * kMatrixSize)] = i + 1;
      input_matrix_b[j + (i * kMatrixSize)] = j + 1;
      true_solution[j + (i * kMatrixSize)] = (i + 1) * (j + 1) * (double)kMatrixSize;
    }
  }
  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix_a.data()));
  task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix_b.data()));
  task_data_mpi->inputs_count.emplace_back(input_matrix_a.size());
  task_data_mpi->inputs_count.emplace_back(input_matrix_b.size());
  task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_matrix_c.data()));
  task_data_mpi->outputs_count.emplace_back(out_matrix_c.size());

  auto test_mpi_task_parallel =
      std::make_shared<deryabin_m_cannons_algorithm_mpi::CannonsAlgorithmMPITaskParallel>(task_data_mpi);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 1;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_mpi_task_parallel);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
  ASSERT_EQ(true_solution, out_matrix_c[0]);
}

TEST(deryabin_m_cannons_algorithm_mpi, test_task_run_Mpi) {
  constexpr size_t kMatrixSize = 256;
  std::vector<double> input_matrix_a1 = std::vector<double>(kMatrixSize * kMatrixSize, 0);
  std::vector<double> input_matrix_b1 = std::vector<double>(kMatrixSize * kMatrixSize, 0);
  std::vector<double> output_matrix_c1 = std::vector<double>(kMatrixSize * kMatrixSize, 0);
  std::vector<std::vector<double>> out_matrix_c1(1, output_matrix_c1);
  std::vector<double> true_solution1 = std::vector<double>(kMatrixSize * kMatrixSize, 0);
  for (unsigned short i = 0; i < kMatrixSize; i++) {
    for (unsigned short j = 0; j < kMatrixSize; j++) {
      input_matrix_a1[j + (i * kMatrixSize)] = i + 1;
      input_matrix_b1[j + (i * kMatrixSize)] = j + 1;
      true_solution1[j + (i * kMatrixSize)] = 2 * (i + 1) * (j + 1) * (double)kMatrixSize;
    }
  }
  auto task_data_par = std::make_shared<ppc::core::TaskData>();
  task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix_a1.data()));
  task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix_b1.data()));
  task_data_par->inputs_count.emplace_back(input_matrix_a1.size());
  task_data_par->inputs_count.emplace_back(input_matrix_b1.size());
  task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_matrix_c1.data()));
  task_data_par->outputs_count.emplace_back(out_matrix_c1.size());

  auto test_task_par =
      std::make_shared<deryabin_m_cannons_algorithm_mpi::CannonsAlgorithmMPITaskParallel>(task_data_par);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 1;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_par);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
  ASSERT_EQ(true_solution1, out_matrix_c1[0]);
}
