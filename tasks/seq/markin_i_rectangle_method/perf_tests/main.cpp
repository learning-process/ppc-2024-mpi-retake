#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "seq/markin_i_rectangle_method/include/ops_seq.hpp"
TEST(markin_i_rectangle_method_seq, rectangle_method_pipeline_run) {
    

  float left = 1;
  float right = 4;
  int steps = 1000;
  float out = 0;




  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&left));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&right));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&steps));
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out));

  

    auto test_task_sequential = std::make_shared<markin_i_rectangle_method_seq::RectangleSequential>(task_data_seq);
    ASSERT_EQ(test_task_sequential->Validation(), true);
    test_task_sequential->PreProcessing();
    test_task_sequential->Run();
    test_task_sequential->PostProcessing();

    auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
    perf_attr->num_running = 10;
    const auto t0 = std::chrono::high_resolution_clock::now();
    perf_attr->current_timer = [&] {
      auto current_time_point = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
      return static_cast<double>(duration) * 1e-9;
    };


  auto perf_results = std::make_shared<ppc::core::PerfResults>();


  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_sequential);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);}
TEST(markin_i_rectangle_method_seq, rectangle_method_task_run) {


  float left = 1;
  float right = 4;
  int steps = 1000;
  float out = 0;



  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&left));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&right));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&steps));
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out));


  auto test_task_sequential = std::make_shared<markin_i_rectangle_method_seq::RectangleSequential>(task_data_seq);
  ASSERT_EQ(test_task_sequential->Validation(), true);
  test_task_sequential->PreProcessing();
  test_task_sequential->Run();
  test_task_sequential->PostProcessing();

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };


  auto perf_results = std::make_shared<ppc::core::PerfResults>();


  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_sequential);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
}