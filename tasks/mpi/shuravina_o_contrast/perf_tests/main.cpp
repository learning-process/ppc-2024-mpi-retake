#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <core/perf/include/perf.hpp>
#include <cstdint>
#include <vector>

#include "mpi/shuravina_o_contrast/include/ops_mpi.hpp"

TEST(shuravina_o_contrast, test_pipeline_run) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  constexpr size_t kSize = 512;
  std::vector<uint8_t> in(kSize * kSize, 128);
  std::vector<uint8_t> out(kSize * kSize, 0);

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data_mpi->inputs_count.emplace_back(in.size());
  task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_mpi->outputs_count.emplace_back(out.size());

  auto test_task_mpi = std::make_shared<shuravina_o_contrast::TestTaskMPI>(task_data_mpi);

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
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  if (world.rank() == 0) {
    for (size_t i = 0; i < out.size(); ++i) {
      EXPECT_EQ(out[i], 255);
    }
  }
}