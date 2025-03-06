#include <gtest/gtest.h>

#include <chrono>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "seq/fomin_v_sobel_edges/include/ops_seq.hpp"

TEST(sequential_sobel_edge_detection_perf_test, test_pipeline_run) {
  // Создание тестового изображения
  const int width = 4;
  const int height = 4;
  std::vector<unsigned char> input_image = {100, 100, 100, 100, 100, 200, 200, 100,
                                            100, 200, 200, 100, 100, 100, 100, 100};
  std::vector<unsigned char> output_image(width * height, 0);

  // Создание TaskData
  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_image.data()));
  task_data_seq->inputs_count.emplace_back(width);
  task_data_seq->inputs_count.emplace_back(height);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_image.data()));
  task_data_seq->outputs_count.emplace_back(width);
  task_data_seq->outputs_count.emplace_back(height);

  // Создание задачи
  auto sobelEdgeDetection = std::make_shared<fomin_v_sobel_edges::SobelEdgeDetection>(task_data_seq);

  // Создание Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Создание и инициализация perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Создание Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(sobelEdgeDetection);
  perfAnalyzer->PipelineRun(perfAttr, perfResults);

  // Вывод результатов производительности
  ppc::core::Perf::PrintPerfStatistic(perfResults);

  // Проверка, что выходное изображение не пустое
  bool is_output_valid = false;
  for (const auto &pixel : output_image) {
    if (pixel != 0) {
      is_output_valid = true;
      break;
    }
  }
  ASSERT_TRUE(is_output_valid);
}

TEST(sequential_sobel_edge_detection_perf_test, test_task_run) {
  // Создание тестового изображения
  const int width = 4;
  const int height = 4;
  std::vector<unsigned char> input_image = {100, 100, 100, 100, 100, 200, 200, 100,
                                            100, 200, 200, 100, 100, 100, 100, 100};
  std::vector<unsigned char> output_image(width * height, 0);

  // Создание TaskData
  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_image.data()));
  task_data_seq->inputs_count.emplace_back(width);
  task_data_seq->inputs_count.emplace_back(height);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_image.data()));
  task_data_seq->outputs_count.emplace_back(width);
  task_data_seq->outputs_count.emplace_back(height);

  // Создание задачи
  auto sobelEdgeDetection = std::make_shared<fomin_v_sobel_edges::SobelEdgeDetection>(task_data_seq);

  // Создание Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Создание и инициализация perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Создание Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(sobelEdgeDetection);
  perfAnalyzer->TaskRun(perfAttr, perfResults);

  // Вывод результатов производительности
  ppc::core::Perf::PrintPerfStatistic(perfResults);

  // Проверка, что выходное изображение не пустое
  bool is_output_valid = false;
  for (const auto &pixel : output_image) {
    if (pixel != 0) {
      is_output_valid = true;
      break;
    }
  }
  ASSERT_TRUE(is_output_valid);
}
