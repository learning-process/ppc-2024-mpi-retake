#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/timer.hpp>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "mpi/dudchenko_o_shtrassen_algorithm/include/ops_mpi.hpp"

namespace {
struct Value {
  double min_value;
  double max_value;
};

std::vector<double> GenerateRandomSquareMatrix(int n, Value value) {
  std::vector<double> matrix(n * n);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dis(value.min_value, value.max_value);

  for (int i = 0; i < n * n; ++i) {
    matrix[i] = dis(gen);
  }
  return matrix;
}
}  // namespace

TEST(dudchenko_o_shtrassen_algorithm_mpi, test_pipeline_run) {
  boost::mpi::communicator world;
  const size_t n = 256;

  std::vector<double> a = GenerateRandomSquareMatrix(n, {.min_value = -50, .max_value = 50});
  std::vector<double> b = GenerateRandomSquareMatrix(n, {.min_value = -50, .max_value = 50});
  std::vector<double> out(n * n, 0.0);

  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
    task_data_par->inputs_count.emplace_back(a.size());
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    task_data_par->inputs_count.emplace_back(b.size());
    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_par->outputs_count.emplace_back(out.size());
  }

  auto test_task_parallel =
      std::make_shared<dudchenko_o_shtrassen_algorithm_mpi::StrassenAlgoriphmParallel>(task_data_par);
  ASSERT_TRUE(test_task_parallel->ValidationImpl());
  ASSERT_TRUE(test_task_parallel->PreProcessingImpl());
  ASSERT_TRUE(test_task_parallel->RunImpl());
  ASSERT_TRUE(test_task_parallel->PostProcessingImpl());

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 2;
  const boost::mpi::timer current_timer;
  perf_attr->current_timer = [&] { return current_timer.elapsed(); };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_parallel);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
  }
}

TEST(dudchenko_o_shtrassen_algorithm_mpi, test_task_run) {
  boost::mpi::communicator world;
  const size_t n = 256;

  std::vector<double> a = GenerateRandomSquareMatrix(n, {.min_value = -50, .max_value = 50});
  std::vector<double> b = GenerateRandomSquareMatrix(n, {.min_value = -50, .max_value = 50});
  std::vector<double> out(n * n, 0.0);

  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
    task_data_par->inputs_count.emplace_back(a.size());
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    task_data_par->inputs_count.emplace_back(b.size());
    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_par->outputs_count.emplace_back(out.size());
  }

  auto test_task_parallel =
      std::make_shared<dudchenko_o_shtrassen_algorithm_mpi::StrassenAlgoriphmParallel>(task_data_par);
  ASSERT_TRUE(test_task_parallel->ValidationImpl());
  ASSERT_TRUE(test_task_parallel->PreProcessingImpl());
  ASSERT_TRUE(test_task_parallel->RunImpl());
  ASSERT_TRUE(test_task_parallel->PostProcessingImpl());

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 2;
  const boost::mpi::timer current_timer;
  perf_attr->current_timer = [&] { return current_timer.elapsed(); };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_parallel);
  perf_analyzer->TaskRun(perf_attr, perf_results);

  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
  }
}