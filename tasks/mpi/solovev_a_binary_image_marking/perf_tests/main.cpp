#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <chrono>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"

#include "mpi/solovev_a_binary_image_marking/include/ops_mpi.hpp"

TEST(solovev_a_binary_image_marking_mpi, pipeline_run) {
  boost::mpi::communicator world;
  const int m = 2500;
  const int n = 2500;

  std::vector<int> data(m * n, 1);
  std::vector<int> labledImage(m * n);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(&m)));
  taskDataPar->inputs_count.emplace_back(1);

  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(&n)));
  taskDataPar->inputs_count.emplace_back(1);

  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(data.data())));
  taskDataPar->inputs_count.emplace_back(data.size());

  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(labledImage.data()));
  taskDataPar->outputs_count.emplace_back(labledImage.size());

  auto TaskParallel = std::make_shared<solovev_a_binary_image_marking::TestMPITaskParallel>(taskDataPar);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(TaskParallel);
  perfAnalyzer->PipelineRun(perfAttr, perfResults);
  ppc::core::Perf::PrintPerfStatistic(perfResults);

  if (world.rank() == 0) {
    for (size_t i = 0; i < labledImage.size(); ++i) {
      ASSERT_EQ(data[i], labledImage[i]);
    }
  }
}

TEST(solovev_a_binary_image_marking_mpi, task_run) {
  boost::mpi::communicator world;
  const int m = 2500;
  const int n = 2500;

  std::vector<int> data(m * n, 1);
  std::vector<int> labledImage(m * n);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(&m)));
  taskDataPar->inputs_count.emplace_back(1);

  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(&n)));
  taskDataPar->inputs_count.emplace_back(1);

  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(data.data())));
  taskDataPar->inputs_count.emplace_back(data.size());

  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(labledImage.data()));
  taskDataPar->outputs_count.emplace_back(labledImage.size());

  auto TaskParallel = std::make_shared<solovev_a_binary_image_marking::TestMPITaskParallel>(taskDataPar);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(TaskParallel);
  perfAnalyzer->TaskRun(perfAttr, perfResults);
  ppc::core::Perf::PrintPerfStatistic(perfResults);

  if (world.rank() == 0) {
    for (size_t i = 0; i < labledImage.size(); ++i) {
      ASSERT_EQ(data[i], labledImage[i]);
    }
  }
}