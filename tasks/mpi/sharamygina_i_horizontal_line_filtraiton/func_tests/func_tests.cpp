#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <cstdint>
#include <limits>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "mpi/sharamygina_i_horizontal_line_filtraiton/include/ops_mpi.h"

namespace sharamygina_i_horizontal_line_filtration_mpi {
namespace {
std::vector<unsigned int> GetImage(int rows, int cols) {
  std::vector<unsigned int> temporary_im(rows * cols);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<unsigned int> dist(0, std::numeric_limits<unsigned int>::max());
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      temporary_im[(i * cols) + j] = dist(gen);
    }
  }
  return temporary_im;
}

unsigned int ApplyGaussianFilter(const std::vector<unsigned int>& image, int x, int y, int rows, int cols) {
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

std::vector<unsigned int> ToFiltSeq(const std::vector<unsigned int>& image, int rows, int cols) {
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

void InitializeTaskData(std::shared_ptr<ppc::core::TaskData>& task_data, std::vector<unsigned int>& image, int rows,
                        int cols, std::vector<unsigned int>& received_image,
                        std::vector<unsigned int>& expected_image) {
  image = sharamygina_i_horizontal_line_filtration_mpi::GetImage(rows, cols);
  expected_image = sharamygina_i_horizontal_line_filtration_mpi::ToFiltSeq(image, rows, cols);

  task_data->inputs_count.emplace_back(rows);
  task_data->inputs_count.emplace_back(cols);
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(image.data()));

  received_image.resize(rows * cols);
  task_data->outputs_count.emplace_back(received_image.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(received_image.data()));
}
void CheckResults(const std::vector<unsigned int>& received_image, const std::vector<unsigned int>& expected_image) {
  for (unsigned int i = 0; i < received_image.size(); i++) {
    ASSERT_EQ(received_image[i], expected_image[i]) << "Difference at i=" << i;
  }
}
}  // namespace
}  // namespace sharamygina_i_horizontal_line_filtration_mpi

TEST(sharamygina_i_horizontal_line_filtration_mpi, SampleImageTest) {
  boost::mpi::communicator world;

  int rows = 4;
  int cols = 4;

  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();

  std::vector<unsigned int> received_image;
  std::vector<unsigned int> image(rows * cols);
  std::vector<unsigned int> expected_image(rows * cols);

  if (world.rank() == 0) {
    sharamygina_i_horizontal_line_filtration_mpi::InitializeTaskData(task_data, image, rows, cols, received_image,
                                                                     expected_image);
  }

  sharamygina_i_horizontal_line_filtration_mpi::HorizontalLineFiltrationMpi test_task(task_data);

  ASSERT_TRUE(test_task.ValidationImpl());
  ASSERT_TRUE(test_task.PreProcessingImpl());
  ASSERT_TRUE(test_task.RunImpl());
  ASSERT_TRUE(test_task.PostProcessingImpl());

  if (world.rank() == 0) {
    sharamygina_i_horizontal_line_filtration_mpi::CheckResults(received_image, expected_image);
  }
}

TEST(sharamygina_i_horizontal_line_filtration_mpi, BigImageTest) {
  boost::mpi::communicator world;

  int rows = 300;
  int cols = 300;

  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();

  std::vector<unsigned int> received_image;
  std::vector<unsigned int> image(rows * cols);
  std::vector<unsigned int> expected_image(rows * cols);

  if (world.rank() == 0) {
    sharamygina_i_horizontal_line_filtration_mpi::InitializeTaskData(task_data, image, rows, cols, received_image,
                                                                     expected_image);
  }

  sharamygina_i_horizontal_line_filtration_mpi::HorizontalLineFiltrationMpi test_task(task_data);
  ASSERT_TRUE(test_task.ValidationImpl());
  ASSERT_TRUE(test_task.PreProcessingImpl());
  ASSERT_TRUE(test_task.RunImpl());
  ASSERT_TRUE(test_task.PostProcessingImpl());

  if (world.rank() == 0) {
    sharamygina_i_horizontal_line_filtration_mpi::CheckResults(received_image, expected_image);
  }
}

TEST(sharamygina_i_horizontal_line_filtration_mpi, SmallImageTest) {
  boost::mpi::communicator world;

  int rows = 4;
  int cols = 5;

  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();

  std::vector<unsigned int> received_image;
  std::vector<unsigned int> image(rows * cols);
  std::vector<unsigned int> expected_image(rows * cols);

  if (world.rank() == 0) {
    sharamygina_i_horizontal_line_filtration_mpi::InitializeTaskData(task_data, image, rows, cols, received_image,
                                                                     expected_image);
  }

  sharamygina_i_horizontal_line_filtration_mpi::HorizontalLineFiltrationMpi test_task(task_data);

  ASSERT_TRUE(test_task.ValidationImpl());
  ASSERT_TRUE(test_task.PreProcessingImpl());
  ASSERT_TRUE(test_task.RunImpl());
  ASSERT_TRUE(test_task.PostProcessingImpl());

  if (world.rank() == 0) {
    sharamygina_i_horizontal_line_filtration_mpi::CheckResults(received_image, expected_image);
  }
}

TEST(sharamygina_i_horizontal_line_filtration_mpi, SquareImageTest) {
  boost::mpi::communicator world;

  int rows = 13;
  int cols = 13;

  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();

  std::vector<unsigned int> received_image;
  std::vector<unsigned int> image(rows * cols);
  std::vector<unsigned int> expected_image(rows * cols);

  if (world.rank() == 0) {
    sharamygina_i_horizontal_line_filtration_mpi::InitializeTaskData(task_data, image, rows, cols, received_image,
                                                                     expected_image);
  }

  sharamygina_i_horizontal_line_filtration_mpi::HorizontalLineFiltrationMpi test_task(task_data);

  ASSERT_TRUE(test_task.ValidationImpl());
  ASSERT_TRUE(test_task.PreProcessingImpl());
  ASSERT_TRUE(test_task.RunImpl());
  ASSERT_TRUE(test_task.PostProcessingImpl());

  if (world.rank() == 0) {
    sharamygina_i_horizontal_line_filtration_mpi::CheckResults(received_image, expected_image);
  }
}

TEST(sharamygina_i_horizontal_line_filtration_mpi, HorizontalImageTest) {
  boost::mpi::communicator world;

  int rows = 9;
  int cols = 20;

  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();

  std::vector<unsigned int> received_image;
  std::vector<unsigned int> image(rows * cols);
  std::vector<unsigned int> expected_image(rows * cols);

  if (world.rank() == 0) {
    sharamygina_i_horizontal_line_filtration_mpi::InitializeTaskData(task_data, image, rows, cols, received_image,
                                                                     expected_image);
  }

  sharamygina_i_horizontal_line_filtration_mpi::HorizontalLineFiltrationMpi test_task(task_data);

  ASSERT_TRUE(test_task.ValidationImpl());
  ASSERT_TRUE(test_task.PreProcessingImpl());
  ASSERT_TRUE(test_task.RunImpl());
  ASSERT_TRUE(test_task.PostProcessingImpl());

  if (world.rank() == 0) {
    sharamygina_i_horizontal_line_filtration_mpi::CheckResults(received_image, expected_image);
  }
}

TEST(sharamygina_i_horizontal_line_filtration_mpi, VerticalImageTest) {
  boost::mpi::communicator world;

  int rows = 12;
  int cols = 5;

  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();

  std::vector<unsigned int> received_image;
  std::vector<unsigned int> image(rows * cols);
  std::vector<unsigned int> expected_image(rows * cols);

  if (world.rank() == 0) {
    sharamygina_i_horizontal_line_filtration_mpi::InitializeTaskData(task_data, image, rows, cols, received_image,
                                                                     expected_image);
  }

  sharamygina_i_horizontal_line_filtration_mpi::HorizontalLineFiltrationMpi test_task(task_data);

  ASSERT_TRUE(test_task.ValidationImpl());
  ASSERT_TRUE(test_task.PreProcessingImpl());
  ASSERT_TRUE(test_task.RunImpl());
  ASSERT_TRUE(test_task.PostProcessingImpl());

  if (world.rank() == 0) {
    sharamygina_i_horizontal_line_filtration_mpi::CheckResults(received_image, expected_image);
  }
}