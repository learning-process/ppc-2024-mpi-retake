#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

#include "boost/mpi/communicator.hpp"
#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "mpi/shkurinskaya_e_fox_matrix_mult/include/ops_sec.hpp"

TEST(shkurinskaya_e_fox_mat_mul_mpi, test_pipeline_run) {
  boost::mpi::communicator world;
  int root = (int)sqrt(world.size());
  if (world.size() != (int)pow(root, 2)) {
    GTEST_SKIP();
  }

  int matrix_size = 576;
  std::vector<double> in1(matrix_size * matrix_size, 0.0);
  std::vector<double> in2(matrix_size * matrix_size, 0.0);
  std::vector<double> out(matrix_size * matrix_size);
  std::vector<double> ans(matrix_size * matrix_size, 0.0);
  for (int i = 0; i < matrix_size; ++i) {
    in1[(i * matrix_size) + i] = 1;
    in2[(i * matrix_size) + i] = 1;
    ans[(i * matrix_size) + i] = 1;
  }

  // Create task_data
  auto mpi_data = std::make_shared<ppc::core::TaskData>();
  mpi_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in1.data()));
  mpi_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in2.data()));
  mpi_data->inputs_count.emplace_back(matrix_size);
  mpi_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  mpi_data->outputs_count.emplace_back(matrix_size);

  // Create Task
  auto job = std::make_shared<shkurinskaya_e_fox_mat_mul_mpi::FoxMatMulMPI>(mpi_data);

  // Create Perf attributes
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 1;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Create and init perf results
  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(job);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  // Create Perf analyzer
  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
    for (int i = 0; i < (int)ans.size(); ++i) {
      ASSERT_NEAR(ans[i], out[i], 1);
    }
  }
}

TEST(shkurinskaya_e_fox_mat_mul_mpi, test_task_run) {
  boost::mpi::communicator world;
  int root = (int)sqrt(world.size());
  if (world.size() != (int)pow(root, 2)) {
    GTEST_SKIP();
  }

  int matrix_size = 576;
  std::vector<double> in1(matrix_size * matrix_size, 0.0);
  std::vector<double> in2(matrix_size * matrix_size, 0.0);
  std::vector<double> out(matrix_size * matrix_size);
  std::vector<double> ans(matrix_size * matrix_size, 0.0);
  for (int i = 0; i < matrix_size; ++i) {
    in1[(i * matrix_size) + i] = 1;
    in2[(i * matrix_size) + i] = 1;
    ans[(i * matrix_size) + i] = 1;
  }

  // Create task_data
  auto mpi_data = std::make_shared<ppc::core::TaskData>();
  mpi_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in1.data()));
  mpi_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in2.data()));
  mpi_data->inputs_count.emplace_back(matrix_size);
  mpi_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  mpi_data->outputs_count.emplace_back(matrix_size);

  // Create Task
  auto job = std::make_shared<shkurinskaya_e_fox_mat_mul_mpi::FoxMatMulMPI>(mpi_data);

  // Create Perf attributes
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 1;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Create and init perf results
  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(job);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  // Create Perf analyzer
  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
    for (int i = 0; i < (int)ans.size(); ++i) {
      ASSERT_NEAR(ans[i], out[i], 1);
    }
  }
}
