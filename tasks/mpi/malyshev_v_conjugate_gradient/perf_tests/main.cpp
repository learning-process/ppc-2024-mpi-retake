// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <memory>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "mpi/malyshev_v_conjugate_gradient/include/ops_mpi.hpp"

namespace malyshev_v_conjugate_gradient_mpi {
namespace {
void GenerateData(size_t size, std::vector<double>& matrix_a, std::vector<double>& vector_b,
                  std::vector<double>& expected) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<> dist(-5.0, 5.0);
  std::vector<double> m(size * size);
  for (size_t i = 0; i < size; i++) {
    for (size_t j = 0; j < size; j++) {
      m[(i * size) + j] = dist(gen);
    }
  }
  matrix_a.assign(size * size, 0.0);
  for (size_t i = 0; i < size; i++) {
    for (size_t j = 0; j < size; j++) {
      for (size_t k = 0; k < size; k++) {
        matrix_a[(i * size) + j] += m[(k * size) + i] * m[(k * size) + j];
      }
    }
  }
  for (size_t i = 0; i < size; i++) {
    matrix_a[(i * size) + i] += static_cast<double>(size);
  }
  expected.resize(size);
  for (size_t i = 0; i < size; i++) {
    expected[i] = dist(gen);
  }
  vector_b.assign(size, 0.0);
  for (size_t i = 0; i < size; i++) {
    for (size_t j = 0; j < size; j++) {
      vector_b[i] += matrix_a[(i * size) + j] * expected[j];
    }
  }
}
}  // namespace
}  // namespace malyshev_v_conjugate_gradient_mpi

TEST(malyshev_v_conjugate_gradient_mpi, test_pipeline_run) {
  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();
  boost::mpi::communicator world;
  int size = 800;
  double epsilon = 1e-5;
  std::vector<double> matrix_a;
  std::vector<double> vector_b;
  std::vector<double> expected;
  std::vector<double> out(size, 0.0);

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    malyshev_v_conjugate_gradient_mpi::GenerateData(size, matrix_a, vector_b, expected);
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix_a.data()));
    task_data_mpi->inputs_count.emplace_back(out.size());
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(vector_b.data()));
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    task_data_mpi->outputs_count.emplace_back(out.size());
  }

  auto test_task_mpi = std::make_shared<malyshev_v_conjugate_gradient_mpi::ConjugateGradientMethod>(task_data_mpi);
  ASSERT_EQ(test_task_mpi->Validation(), true);
  test_task_mpi->PreProcessing();
  test_task_mpi->Run();
  test_task_mpi->PostProcessing();

  if (world.rank() == 0) {
    for (size_t i = 0; i < expected.size(); ++i) {
      ASSERT_NEAR(expected[i], out[i], 1e-3);
    }
  }

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_mpi);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
  }
}

TEST(malyshev_v_conjugate_gradient_mpi, test_task_run) {
  boost::mpi::communicator world;
  int size = 800;
  double epsilon = 1e-5;
  std::vector<double> matrix_a;
  std::vector<double> vector_b;
  std::vector<double> expected;
  std::vector<double> out(size, 0.0);
  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    malyshev_v_conjugate_gradient_mpi::GenerateData(size, matrix_a, vector_b, expected);
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix_a.data()));
    task_data_mpi->inputs_count.emplace_back(out.size());
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(vector_b.data()));
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    task_data_mpi->outputs_count.emplace_back(out.size());
  }
  auto test_task_mpi = std::make_shared<malyshev_v_conjugate_gradient_mpi::ConjugateGradientMethod>(task_data_mpi);
  ASSERT_EQ(test_task_mpi->Validation(), true);
  test_task_mpi->PreProcessing();
  test_task_mpi->Run();
  test_task_mpi->PostProcessing();

  if (world.rank() == 0) {
    for (size_t i = 0; i < expected.size(); ++i) {
      ASSERT_NEAR(expected[i], out[i], 1e-3);
    }
  }

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };
  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_mpi);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
  }
}