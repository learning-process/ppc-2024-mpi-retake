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
std::vector<unsigned int> GetImage(int k_rows, int k_cols) {
  std::vector<unsigned int> temporary_im(k_rows * k_cols);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<unsigned int> dist(0, std::numeric_limits<unsigned int>::max());
  for (int i = 0; i < k_rows; i++) {
    for (int j = 0; j < k_cols; j++) {
      temporary_im[(i * k_cols) + j] = dist(gen);
    }
  }
  return temporary_im;
}

unsigned int ApplyGaussianFilter(const std::vector<unsigned int> &image, int x, int y, int rows, int cols) {
  unsigned int gauss[3][3]{{1, 2, 1}, {2, 4, 2}, {1, 2, 1}};
  unsigned int sum = 0;

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      int t_x = x + i - 1;
      int t_y = y + j - 1;

      if (t_x < 0 || t_x >= rows) {
        t_x = x;
      }
      if (t_y < 0 || t_y >= cols) {
        t_y = y;
      }

      sum += static_cast<unsigned int>(image[(t_x * cols) + t_y] * gauss[i][j]);
    }
  }

  return sum / 16;
}

std::vector<unsigned int> ToFiltSeq(const std::vector<unsigned int> &image, int rows, int cols) {
  std::vector<unsigned int> final_image(rows * cols, 0);

  for (int x = 0; x < rows; x++) {
    for (int y = 0; y < cols; y++) {
      if (x < 1 || x >= rows - 1 || y < 1 || y >= cols - 1) {
        final_image[(x * cols) + y] = 0;
      } else {
        final_image[(x * cols) + y] = ApplyGaussianFilter(image, x, y, rows, cols);
      }
    }
  }
  return final_image;
}
}  // namespace
}  // namespace sharamygina_i_horizontal_line_filtration_mpi

TEST(sharamygina_i_horizontal_line_filtraiton_mpi, test_pipeline_run) {
  boost::mpi::communicator world;

  int k_rows = 6000;
  int k_cols = 6000;

  // Create data
  std::vector<unsigned int> received_image;
  std::vector<unsigned int> image(k_rows * k_cols);
  std::vector<unsigned int> expected_image(k_rows * k_cols);
  image = sharamygina_i_horizontal_line_filtration_mpi::GetImage(k_rows, k_cols);
  expected_image = sharamygina_i_horizontal_line_filtration_mpi::ToFiltSeq(image, k_rows, k_cols);

  // Create task_data
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    image = sharamygina_i_horizontal_line_filtration_mpi::GetImage(k_rows, k_cols);
    expected_image = sharamygina_i_horizontal_line_filtration_mpi::ToFiltSeq(image, k_rows, k_cols);

    task_data->inputs_count.emplace_back(k_rows);
    task_data->inputs_count.emplace_back(k_cols);

    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(image.data()));

    received_image.resize(k_rows * k_cols);
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

  int k_rows = 6000;
  int k_cols = 6000;

  // Create data
  std::vector<unsigned int> received_image;
  std::vector<unsigned int> image(k_rows * k_cols);
  std::vector<unsigned int> expected_image(k_rows * k_cols);
  image = sharamygina_i_horizontal_line_filtration_mpi::GetImage(k_rows, k_cols);
  expected_image = sharamygina_i_horizontal_line_filtration_mpi::ToFiltSeq(image, k_rows, k_cols);

  // Create task_data
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    image = sharamygina_i_horizontal_line_filtration_mpi::GetImage(k_rows, k_cols);
    expected_image = sharamygina_i_horizontal_line_filtration_mpi::ToFiltSeq(image, k_rows, k_cols);

    task_data->inputs_count.emplace_back(k_rows);
    task_data->inputs_count.emplace_back(k_cols);

    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(image.data()));

    received_image.resize(k_rows * k_cols);
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
