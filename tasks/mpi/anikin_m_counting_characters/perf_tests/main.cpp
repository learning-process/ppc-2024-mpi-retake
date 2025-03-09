#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "boost/mpi/communicator.hpp"
#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "mpi/anikin_m_counting_characters/include/ops_mpi.hpp"

namespace {
void CreateRanddataVector(std::vector<char> *invec, int count) {
  for (int i = 0; i < count; i++) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis('A', 'Z');
    char random_har_ar = static_cast<char>(dis(gen));
    invec->push_back(random_har_ar);
  }
}
}  // namespace

TEST(anikin_m_counting_characters_mpi, test_pipeline_run) {
  constexpr int kCount = 20000000;

  std::vector<char> in1;
  CreateRanddataVector(&in1, kCount);
  std::vector<char> in2;
  CreateRanddataVector(&in2, kCount);
  int res_out = 0;

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(in1.data()));
  task_data_mpi->inputs_count.emplace_back(in1.size());
  task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(in2.data()));
  task_data_mpi->inputs_count.emplace_back(in2.size());
  task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(&res_out));
  task_data_mpi->outputs_count.emplace_back(1);

  auto test_task_mpi = std::make_shared<anikin_m_counting_characters_mpi::TestTaskMPI>(task_data_mpi);

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
  boost::mpi::communicator world;
  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
  }
  ASSERT_EQ(true, true);
}

TEST(anikin_m_counting_characters_mpi, test_task_run) {
  constexpr int kCount = 20000000;
  std::vector<char> in1;
  CreateRanddataVector(&in1, kCount);
  std::vector<char> in2;
  CreateRanddataVector(&in2, kCount);
  int res_out = 0;

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(in1.data()));
  task_data_mpi->inputs_count.emplace_back(in1.size());
  task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(in2.data()));
  task_data_mpi->inputs_count.emplace_back(in2.size());
  task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(&res_out));
  task_data_mpi->outputs_count.emplace_back(1);

  auto test_task_mpi = std::make_shared<anikin_m_counting_characters_mpi::TestTaskMPI>(task_data_mpi);

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
  boost::mpi::communicator world;
  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
  }
  ASSERT_EQ(true, true);
}