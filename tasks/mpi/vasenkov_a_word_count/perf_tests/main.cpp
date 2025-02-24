#include <gtest/gtest.h>

#include <chrono>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "boost/mpi/communicator.hpp"
#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "mpi/vasenkov_a_word_count/include/ops_mpi.hpp"

TEST(vasenkov_a_word_count_mpi, test_pipeline_run) {
  std::string input = "Hello. world! World";
  std::vector<uint8_t> in(input.begin(), input.end());
  std::vector<int> out(1, 0);
  std::vector<int> expect = {3};

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_mpi->inputs_count.emplace_back(in.size());
  task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_mpi->outputs_count.emplace_back(out.size());

  auto test_task_mpi = std::make_shared<vasenkov_a_word_count_mpi::WordCountMPI>(task_data_mpi);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 5000;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_mpi);
  perf_analyzer->PipelineRun(perf_attr, perf_results);

  boost::mpi::communicator world;
  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
  }

  ASSERT_EQ(expect, out);
}

TEST(vasenkov_a_word_count_mpi, test_task_run) {
  std::string input = "Hello. world! World";
  std::vector<uint8_t> in(input.begin(), input.end());
  std::vector<int> out(1, 0);
  std::vector<int> expect = {3};

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_mpi->inputs_count.emplace_back(in.size());
  task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_mpi->outputs_count.emplace_back(out.size());

  auto test_task_mpi = std::make_shared<vasenkov_a_word_count_mpi::WordCountMPI>(task_data_mpi);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 5000;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_mpi);
  perf_analyzer->TaskRun(perf_attr, perf_results);

  boost::mpi::communicator world;
  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
  }

  ASSERT_EQ(expect, out);
}
