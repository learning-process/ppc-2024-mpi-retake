#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <chrono>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "mpi/malyshev_v_lent_horizontal/include/ops_mpi.hpp"

namespace {
std::vector<int> GetRandomMatrix(int rows, int cols) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> matrix(rows * cols);
  for (int i = 0; i < rows * cols; i++) {
    matrix[i] = static_cast<int>(gen() % 100);
  }
  return matrix;
}

std::vector<int> GetRandomVector(int size) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vector(size);
  for (int i = 0; i < size; i++) {
    vector[i] = static_cast<int>(gen() % 100);
  }
  return vector;
}
}  // namespace

TEST(malyshev_v_lent_horizontal_mpi, test_pipeline_Run) {
  boost::mpi::communicator world;

  int cols = 12000;
  int rows = 12000;

  std::vector<int> matrix = GetRandomMatrix(rows, cols);
  std::vector<int> vector = GetRandomVector(cols);
  std::vector<int> out(rows, 0);

  std::vector<int> expect(rows, 0);
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      expect[i] += matrix[(i * cols) + j] * vector[j];
    }
  }

  auto task_data_par = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    task_data_par->inputs_count.emplace_back(matrix.size());
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(vector.data()));
    task_data_par->inputs_count.emplace_back(vector.size());
    task_data_par->inputs_count.emplace_back(rows);
    task_data_par->inputs_count.emplace_back(cols);
    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_par->outputs_count.emplace_back(out.size());
  }

  auto test_task_par = std::make_shared<malyshev_v_lent_horizontal_mpi::MatVecMultMpi>(task_data_par);
  ASSERT_EQ(test_task_par->ValidationImpl(), true);
  test_task_par->PreProcessingImpl();
  test_task_par->RunImpl();
  test_task_par->PostProcessingImpl();

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_par);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
    ASSERT_EQ(expect, out);
  }
}

TEST(malyshev_v_lent_horizontal_mpi, test_task_Run) {
  boost::mpi::communicator world;

  int cols = 12000;
  int rows = 12000;

  std::vector<int> matrix = GetRandomMatrix(rows, cols);
  std::vector<int> vector = GetRandomVector(cols);
  std::vector<int> out(rows, 0);

  std::vector<int> expect(rows, 0);
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      expect[i] += matrix[(i * cols) + j] * vector[j];
    }
  }

  auto task_data_par = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    task_data_par->inputs_count.emplace_back(matrix.size());
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(vector.data()));
    task_data_par->inputs_count.emplace_back(vector.size());
    task_data_par->inputs_count.emplace_back(rows);
    task_data_par->inputs_count.emplace_back(cols);
    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_par->outputs_count.emplace_back(out.size());
  }

  auto test_task_par = std::make_shared<malyshev_v_lent_horizontal_mpi::MatVecMultMpi>(task_data_par);
  ASSERT_EQ(test_task_par->ValidationImpl(), true);
  test_task_par->PreProcessingImpl();
  test_task_par->RunImpl();
  test_task_par->PostProcessingImpl();

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_par);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
    ASSERT_EQ(expect, out);
  }
}