#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "boost/mpi/communicator.hpp"
#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "mpi/fomin_v_sobel_edges/include/ops_mpi.hpp"

TEST(mpi_sobel_edge_detection_perf_test, test_pipeline_run) {
  boost::mpi::communicator world;
  std::vector<unsigned char> global_image;
  std::vector<unsigned char> global_output_image;

  // Создание тестового изображения (например, 8x8)
  const int width = 8;
  const int height = 8;
  global_image.resize(width * height, 100);
  for (int i = 2; i < 6; ++i) {
    for (int j = 2; j < 6; ++j) {
      global_image[i * width + j] = 200;
    }
  }

  // Создание TaskData для параллельной версии
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    global_output_image.resize(width * height, 0);
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_image.data()));
    task_data->inputs_count.emplace_back(width);
    task_data->inputs_count.emplace_back(height);
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_output_image.data()));
    task_data->outputs_count.emplace_back(width);
    task_data->outputs_count.emplace_back(height);
  }

  // Создание и выполнение параллельной задачи
  auto sobelEdgeDetectionMPI = std::make_shared<fomin_v_sobel_edges::SobelEdgeDetectionMPI>(task_data);
  ASSERT_EQ(sobelEdgeDetectionMPI->ValidationImpl(), true);
  sobelEdgeDetectionMPI->PreProcessingImpl();
  sobelEdgeDetectionMPI->RunImpl();
  sobelEdgeDetectionMPI->PostProcessingImpl();

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
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(sobelEdgeDetectionMPI);
  perfAnalyzer->PipelineRun(perfAttr, perfResults);

  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perfResults);

    // Проверка, что выходное изображение не пустое
    bool is_output_valid = false;
    for (const auto &pixel : global_output_image) {
      if (pixel != 0) {
        is_output_valid = true;
        break;
      }
    }
    ASSERT_TRUE(is_output_valid);
  }
}

TEST(mpi_sobel_edge_detection_perf_test, test_task_run) {
  boost::mpi::communicator world;
  std::vector<unsigned char> global_image;
  std::vector<unsigned char> global_output_image;

  // Создание тестового изображения
  const int width = 8;
  const int height = 8;
  global_image.resize(width * height, 100);
  for (int i = 2; i < 6; ++i) {
    for (int j = 2; j < 6; ++j) {
      global_image[i * width + j] = 200;
    }
  }

  // Создание TaskData для параллельной версии
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    global_output_image.resize(width * height, 0);
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_image.data()));
    task_data->inputs_count.emplace_back(width);
    task_data->inputs_count.emplace_back(height);
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_output_image.data()));
    task_data->outputs_count.emplace_back(width);
    task_data->outputs_count.emplace_back(height);
  }

  // Создание и выполнение параллельной задачи
  auto sobelEdgeDetectionMPI = std::make_shared<fomin_v_sobel_edges::SobelEdgeDetectionMPI>(task_data);
  ASSERT_EQ(sobelEdgeDetectionMPI->ValidationImpl(), true);
  sobelEdgeDetectionMPI->PreProcessingImpl();
  sobelEdgeDetectionMPI->RunImpl();
  sobelEdgeDetectionMPI->PostProcessingImpl();

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
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(sobelEdgeDetectionMPI);
  perfAnalyzer->TaskRun(perfAttr, perfResults);

  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perfResults);

    // Проверка, что выходное изображение не пустое
    bool is_output_valid = false;
    for (const auto &pixel : global_output_image) {
      if (pixel != 0) {
        is_output_valid = true;
        break;
      }
    }
    ASSERT_TRUE(is_output_valid);
  }
}