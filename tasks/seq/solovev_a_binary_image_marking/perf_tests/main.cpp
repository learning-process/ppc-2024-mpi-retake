
#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "seq/solovev_a_binary_image_marking/include/ops_sec.hpp"

TEST(solovev_a_binary_image_marking_seq, pipeline_run) {
  const int m = 2500;
  const int n = 2500;

  std::vector<int> data(m * n, 1);
  std::vector<int> labledImage(m * n, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(&m)));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(&n)));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(data.data())));
  taskDataSeq->inputs_count.emplace_back(data.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(labledImage.data()));
  taskDataSeq->outputs_count.emplace_back(labledImage.size());

  auto TaskSequential = std::make_shared<solovev_a_binary_image_marking::TestTaskSequential>(taskDataSeq);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(TaskSequential);
  perfAnalyzer->PipelineRun(perfAttr, perfResults);
  ppc::core::Perf::PrintPerfStatistic(perfResults);

  for (size_t i = 0; i < labledImage.size(); ++i) {
    ASSERT_EQ(data[i], labledImage[i]);
  }
}

TEST(solovev_a_binary_image_marking_seq, task_run) {
  const int m = 2500;
  const int n = 2500;

  std::vector<int> data(m * n, 1);
  std::vector<int> labledImage(m * n, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(&m)));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(&n)));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(data.data())));
  taskDataSeq->inputs_count.emplace_back(data.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(labledImage.data()));
  taskDataSeq->outputs_count.emplace_back(labledImage.size());

  auto TaskSequential = std::make_shared<solovev_a_binary_image_marking::TestTaskSequential>(taskDataSeq);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(TaskSequential);
  perfAnalyzer->TaskRun(perfAttr, perfResults);
  ppc::core::Perf::PrintPerfStatistic(perfResults);

  for (size_t i = 0; i < labledImage.size(); ++i) {
    ASSERT_EQ(data[i], labledImage[i]);
  }
}