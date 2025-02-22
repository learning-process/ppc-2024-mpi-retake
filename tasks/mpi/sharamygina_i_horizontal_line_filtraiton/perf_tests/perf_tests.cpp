#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/timer.hpp>
#include <chrono>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "mpi/sharamygina_i_horizontal_line_filtraiton/include/ops_mpi.h"

namespace sharamygina_i_horizontal_line_filtration_mpi {
std::vector<unsigned int> GetImage(int rows, int cols) {
  std::vector<unsigned int> temporaryIm(rows * cols);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dist(0, std::numeric_limits<unsigned int>::max());
  for (int i = 0; i < rows; i++)
    for (int j = 0; j < cols; j++) temporaryIm[i * cols + j] = dist(gen);
  return temporaryIm;
}

std::vector<unsigned int> ToFiltSeq(const std::vector<unsigned int> &image, int rows, int cols) {  // seq
  std::vector<unsigned int> final_image(rows * cols);
  unsigned int gauss[3][3]{{1, 2, 1}, {2, 4, 2}, {1, 2, 1}};
  for (int x = 0; x < rows; x++)
    for (int y = 0; y < cols; y++) {
      if (x < 1 || x >= rows - 1 || y < 1 || y >= cols - 1) {
        final_image[x * cols + y] = 0;  // ¬озвращаем 0, если x или y наход€тс€ на кра€х матрицы
        continue;
      }
      unsigned int sum = 0;
      for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) {
          int tX = x + i - 1, tY = y + j - 1;
          if (tX < 0 || tX > rows - 1) tX = x;
          if (tY < 0 || tY > cols - 1) tY = y;
          if (tX * cols + tY >= cols * rows) {
            tX = x;
            tY = y;
          }
          sum += static_cast<unsigned int>(image[tX * cols + tY] * (gauss[i][j]));
        }
      final_image[x * cols + y] = sum / 16;
    }
  return final_image;
}

}  // namespace sharamygina_i_horizontal_line_filtration_mpi

TEST(sharamygina_i_horizontal_line_filtraiton_mpi, test_pipeline_run) {
  boost::mpi::communicator world;

  int rows = 5000;
  int cols = 5000;

  // Create data
  std::vector<unsigned int> received_image;
  std::vector<unsigned int> image(rows * cols);
  std::vector<unsigned int> expected_image(rows * cols);
  image = sharamygina_i_horizontal_line_filtration_mpi::GetImage(rows, cols);
  expected_image = sharamygina_i_horizontal_line_filtration_mpi::ToFiltSeq(image, rows, cols);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    image = sharamygina_i_horizontal_line_filtration_mpi::GetImage(rows, cols);
    expected_image = sharamygina_i_horizontal_line_filtration_mpi::ToFiltSeq(image, rows, cols);

    taskData->inputs_count.emplace_back(rows);
    taskData->inputs_count.emplace_back(cols);

    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(image.data()));

    received_image.resize(rows * cols);
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(received_image.data()));
    taskData->outputs_count.emplace_back(received_image.size());
  }

  auto testTask =
      std::make_shared<sharamygina_i_horizontal_line_filtration_mpi::horizontal_line_filtration_mpi>(taskData);

  ASSERT_EQ(testTask->ValidationImpl(), true);
  testTask->PreProcessingImpl();
  testTask->RunImpl();
  testTask->PostProcessingImpl();

  // Create Perf attributes
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const boost::mpi::timer current_timer;
  perf_attr->current_timer = [&] { return current_timer.elapsed(); };
  // Create and init perf results
  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  // Create Perf analyzer
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(testTask);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
    ASSERT_EQ(expected_image, received_image);
  }
}

TEST(sharamygina_i_horizontal_line_filtraiton_mpi, test_task_run) {
  boost::mpi::communicator world;

  int rows = 5000;
  int cols = 5000;

  // Create data
  std::vector<unsigned int> received_image;
  std::vector<unsigned int> image(rows * cols);
  std::vector<unsigned int> expected_image(rows * cols);
  image = sharamygina_i_horizontal_line_filtration_mpi::GetImage(rows, cols);
  expected_image = sharamygina_i_horizontal_line_filtration_mpi::ToFiltSeq(image, rows, cols);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    image = sharamygina_i_horizontal_line_filtration_mpi::GetImage(rows, cols);
    expected_image = sharamygina_i_horizontal_line_filtration_mpi::ToFiltSeq(image, rows, cols);

    taskData->inputs_count.emplace_back(rows);
    taskData->inputs_count.emplace_back(cols);

    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(image.data()));

    received_image.resize(rows * cols);
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(received_image.data()));
    taskData->outputs_count.emplace_back(received_image.size());
  }
  auto testTask =
      std::make_shared<sharamygina_i_horizontal_line_filtration_mpi::horizontal_line_filtration_mpi>(taskData);

  ASSERT_EQ(testTask->ValidationImpl(), true);
  testTask->PreProcessingImpl();
  testTask->RunImpl();
  testTask->PostProcessingImpl();

  // Create Perf attributes
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const boost::mpi::timer current_timer;
  perf_attr->current_timer = [&] { return current_timer.elapsed(); };
  // Create and init perf results
  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  // Create Perf analyzer
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(testTask);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
  }
}
