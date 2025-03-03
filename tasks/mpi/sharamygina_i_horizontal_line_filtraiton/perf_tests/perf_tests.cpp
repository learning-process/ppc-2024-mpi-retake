#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/timer.hpp>
#include <cstdint>
#include <limits>
#include <memory>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "mpi/sharamygina_i_horizontal_line_filtraiton/include/ops_mpi.h"

namespace sharamygina_i_horizontal_line_filtration_mpi {
namespace {
std::vector<unsigned int> GetImage(int kRows, int kCols) {
  std::vector<unsigned int> temporary_im(kRows * kCols);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dist(0, std::numeric_limits<unsigned int>::max());
  for (int i = 0; i < kRows; i++) {
    for (int j = 0; j < kCols; j++) {
      temporary_im[(i * kCols) + j] = dist(gen);
    }
  }
  return temporary_im;
}

std::vector<unsigned int> ToFiltSeq(const std::vector<unsigned int> &image, int kRows, int kCols) {
  std::vector<unsigned int> final_image(kRows * kCols);
  unsigned int gauss[3][3]{{1, 2, 1}, {2, 4, 2}, {1, 2, 1}};
  for (int x = 0; x < kRows; x++) {
    for (int y = 0; y < kCols; y++) {
      if (x < 1 || x >= kRows - 1 || y < 1 || y >= kCols - 1) {
        final_image[(x * kCols) + y] = 0;
        continue;
      }
      unsigned int sum = 0;
      for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
          int tX = x + i - 1;
          int tY = y + j - 1;
          if (tX < 0 || tX > kRows - 1) {
            tX = x;
          }
          if (tY < 0 || tY > kCols - 1) {
            tY = y;
          }
          if (tX * kCols + tY >= kCols * kRows) {
            tX = x;
            tY = y;
          }
          sum += static_cast<unsigned int>(image[(tX * kCols) + tY] * (gauss[i][j]));
        }
      }
      final_image[(x * kCols) + y] = sum / 16;
    }
  }
  return final_image;
}
}  // namespace
}  // namespace sharamygina_i_horizontal_line_filtration_mpi

TEST(sharamygina_i_horizontal_line_filtraiton_mpi, test_pipeline_run) {
  boost::mpi::communicator world;

  int kRows = 6000;
  int kCols = 6000;

  // Create data
  std::vector<unsigned int> received_image;
  std::vector<unsigned int> image(kRows * kCols);
  std::vector<unsigned int> expected_image(kRows * kCols);
  image = sharamygina_i_horizontal_line_filtration_mpi::GetImage(kRows, kCols);
  expected_image = sharamygina_i_horizontal_line_filtration_mpi::ToFiltSeq(image, kRows, kCols);

  // Create task_data
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    image = sharamygina_i_horizontal_line_filtration_mpi::GetImage(kRows, kCols);
    expected_image = sharamygina_i_horizontal_line_filtration_mpi::ToFiltSeq(image, kRows, kCols);

    task_data->inputs_count.emplace_back(kRows);
    task_data->inputs_count.emplace_back(kCols);

    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(image.data()));

    received_image.resize(kRows * kCols);
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(received_image.data()));
    task_data->outputs_count.emplace_back(received_image.size());
  }

  auto test_task =
      std::make_shared<sharamygina_i_horizontal_line_filtration_mpi::HorizontalLineFiltrationMpi>(task_data);

  ASSERT_EQ(test_task->ValidationImpl(), true);
  test_task->PreProcessingImpl();
  test_task->RunImpl();
  test_task->PostProcessingImpl();

  // Create Perf attributes
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const boost::mpi::timer current_timer;
  perf_attr->current_timer = [&] { return current_timer.elapsed(); };
  // Create and init perf results
  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  // Create Perf analyzer
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
    ASSERT_EQ(expected_image, received_image);
  }
}

TEST(sharamygina_i_horizontal_line_filtraiton_mpi, test_task_run) {
  boost::mpi::communicator world;

  int kRows = 6000;
  int kCols = 6000;

  // Create data
  std::vector<unsigned int> received_image;
  std::vector<unsigned int> image(kRows * kCols);
  std::vector<unsigned int> expected_image(kRows * kCols);
  image = sharamygina_i_horizontal_line_filtration_mpi::GetImage(kRows, kCols);
  expected_image = sharamygina_i_horizontal_line_filtration_mpi::ToFiltSeq(image, kRows, kCols);

  // Create task_data
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    image = sharamygina_i_horizontal_line_filtration_mpi::GetImage(kRows, kCols);
    expected_image = sharamygina_i_horizontal_line_filtration_mpi::ToFiltSeq(image, kRows, kCols);

    task_data->inputs_count.emplace_back(kRows);
    task_data->inputs_count.emplace_back(kCols);

    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(image.data()));

    received_image.resize(kRows * kCols);
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(received_image.data()));
    task_data->outputs_count.emplace_back(received_image.size());
  }
  auto test_task =
      std::make_shared<sharamygina_i_horizontal_line_filtration_mpi::HorizontalLineFiltrationMpi>(task_data);

  ASSERT_EQ(test_task->ValidationImpl(), true);
  test_task->PreProcessingImpl();
  test_task->RunImpl();
  test_task->PostProcessingImpl();

  // Create Perf attributes
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const boost::mpi::timer current_timer;
  perf_attr->current_timer = [&] { return current_timer.elapsed(); };
  // Create and init perf results
  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  // Create Perf analyzer
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
  }
}
