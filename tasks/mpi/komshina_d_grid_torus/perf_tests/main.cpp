#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <chrono>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "mpi/komshina_d_grid_torus/include/ops_mpi.hpp"

TEST(komshina_d_grid_torus_mpi, test_pipeline_Run) {
  boost::mpi::communicator world;

  int sqrtN = static_cast<int>(std::sqrt(world.size()));
  bool isSquareGrid = (sqrtN * sqrtN == world.size());

  if (isSquareGrid) {
    std::vector<int> input{5120, 0};
    std::vector<int> expectedPath{0};
    std::vector<int> output(1, 0);
    std::vector<int> outputPath(2 * sqrtN, -1);
    std::shared_ptr<ppc::core::TaskData> task_data_mpi = std::make_shared<ppc::core::TaskData>();

    if (world.rank() == 0) {
      task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
      task_data_mpi->inputs_count.emplace_back(input.size());
      task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
      task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(outputPath.data()));
      task_data_mpi->outputs_count.emplace_back(output.size());
      task_data_mpi->outputs_count.emplace_back(outputPath.size());
    }

    auto test_task_mpi = std::make_shared<komshina_d_grid_torus_mpi::TestTaskMPI>(task_data_mpi);
    ASSERT_EQ(test_task_mpi->ValidationImpl(), true);
    test_task_mpi->PreProcessingImpl();
    test_task_mpi->RunImpl();
    test_task_mpi->PostProcessingImpl();

    // Create Perf attributes
    auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
    perf_attr->num_running = 10;
    const auto t0 = std::chrono::high_resolution_clock::now();
    perf_attr->current_timer = [&] {
      auto current_time_point = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
      return static_cast<double>(duration) * 1e-9;
    };

    // Create and init perf results
    auto perf_results = std::make_shared<ppc::core::PerfResults>();

    // Create Perf analyzer
    auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_mpi);
    perf_analyzer->PipelineRun(perf_attr, perf_results);

    if (world.rank() == 0) {
      ppc::core::Perf::PrintPerfStatistic(perf_results);
      outputPath.erase(std::remove(outputPath.begin(), outputPath.end(), -1), outputPath.end());
      ASSERT_EQ(output[0], input[0]);
      ASSERT_EQ(outputPath, expectedPath);
    }
  }
}

TEST(komshina_d_grid_torus_mpi, test_task_Run) {
  boost::mpi::communicator world;

  int sqrtN = static_cast<int>(std::sqrt(world.size()));
  bool isSquareGrid = (sqrtN * sqrtN == world.size());

  if (isSquareGrid) {
    std::vector<int> input{5120, 0};
    std::vector<int> expectedPath{0};
    std::vector<int> output(1, 0);
    std::vector<int> outputPath(2 * sqrtN, -1);
    std::shared_ptr<ppc::core::TaskData> task_data_mpi = std::make_shared<ppc::core::TaskData>();

    if (world.rank() == 0) {
      task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
      task_data_mpi->inputs_count.emplace_back(input.size());
      task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
      task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(outputPath.data()));
      task_data_mpi->outputs_count.emplace_back(output.size());
      task_data_mpi->outputs_count.emplace_back(outputPath.size());
    }

    auto test_task_mpi = std::make_shared<komshina_d_grid_torus_mpi::TestTaskMPI>(task_data_mpi);
    ASSERT_EQ(test_task_mpi->ValidationImpl(), true);
    test_task_mpi->PreProcessingImpl();
    test_task_mpi->RunImpl();
    test_task_mpi->PostProcessingImpl();

    // Create Perf attributes
    auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
    perf_attr->num_running = 10;
    const auto t0 = std::chrono::high_resolution_clock::now();
    perf_attr->current_timer = [&] {
      auto current_time_point = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
      return static_cast<double>(duration) * 1e-9;
    };

    // Create and init perf results
    auto perf_results = std::make_shared<ppc::core::PerfResults>();

    // Create Perf analyzer
    auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_mpi);
    perf_analyzer->TaskRun(perf_attr, perf_results);

    if (world.rank() == 0) {
      ppc::core::Perf::PrintPerfStatistic(perf_results);
      outputPath.erase(std::remove(outputPath.begin(), outputPath.end(), -1), outputPath.end());
      ASSERT_EQ(output[0], input[0]);
      ASSERT_EQ(outputPath, expectedPath);
    }
  }
}
