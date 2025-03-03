#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <cstdint>
#include <random>
#include <vector>

#include "mpi/sharamygina_i_horizontal_line_filtraiton/include/ops_mpi.h"

namespace sharamygina_i_horizontal_line_filtration_mpi {
namespace {
std::vector<unsigned int> GetImage(int rows, int cols) {
  std::vector<unsigned int> temporary_im(rows * cols);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dist(0, std::numeric_limits<unsigned int>::max());
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      temporary_im[(i * cols) + j] = dist(gen);
    }
  }
  return temporary_im;
}

std::vector<unsigned int> ToFiltSeq(const std::vector<unsigned int>& image, int rows, int cols) {  // seq
  std::vector<unsigned int> final_image(rows * cols);
  unsigned int gauss[3][3]{{1, 2, 1}, {2, 4, 2}, {1, 2, 1}};
  for (int x = 0; x < rows; x++) {
    for (int y = 0; y < cols; y++) {
      if (x < 1 || x >= rows - 1 || y < 1 || y >= cols - 1) {
        final_image[(x * cols) + y] = 0;
        continue;
      }
      unsigned int sum = 0;
      for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
          int tX = x + i - 1;
          int tY = y + j - 1;
          if (tX < 0 || tX > rows - 1) {
            tX = x;
          }
          if (tY < 0 || tY > cols - 1) {
            tY = y;
          }
          if (tX * cols + tY >= cols * rows) {
            tX = x;
            tY = y;
          }
          sum += static_cast<unsigned int>(image[(tX * cols) + tY] * (gauss[i][j]));
        }
      }
      final_image[(x * cols) + y] = sum / 16;
    }
  }
  return final_image;
}
}  // namespace
}  // namespace sharamygina_i_horizontal_line_filtration_mpi

TEST(sharamygina_i_horizontal_line_filtration_mpi, SampleImageTest) {
  boost::mpi::communicator world;
  int rows = 4;
  int cols = 4;

  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();

  std::vector<unsigned int> received_image;
  std::vector<unsigned int> image = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  std::vector<unsigned int> expected_image = {0, 0, 0, 0, 0, 6, 7, 0, 0, 10, 11, 0, 0, 0, 0, 0};

  if (world.rank() == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(image.data()));
    task_data->inputs_count.emplace_back(rows);
    task_data->inputs_count.emplace_back(cols);

    received_image.resize(rows * cols);
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(received_image.data()));
    task_data->outputs_count.emplace_back(received_image.size());
  }

  sharamygina_i_horizontal_line_filtration_mpi::HorizontalLineFiltrationMpi test_task(task_data);
  ASSERT_TRUE(test_task.ValidationImpl());
  ASSERT_TRUE(test_task.PreProcessingImpl());
  ASSERT_TRUE(test_task.RunImpl());
  ASSERT_TRUE(test_task.PostProcessingImpl());

  if (world.rank() == 0) {
    for (unsigned int i = 0; i < received_image.size(); i++) {
      ASSERT_EQ(received_image[i], expected_image[i]) << "Difference at i=" << i;
    }
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
    image = sharamygina_i_horizontal_line_filtration_mpi::GetImage(rows, cols);
    expected_image = sharamygina_i_horizontal_line_filtration_mpi::ToFiltSeq(image, rows, cols);

    task_data->inputs_count.emplace_back(rows);
    task_data->inputs_count.emplace_back(cols);

    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(image.data()));

    received_image.resize(rows * cols);
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(received_image.data()));
    task_data->outputs_count.emplace_back(received_image.size());
  }

  sharamygina_i_horizontal_line_filtration_mpi::HorizontalLineFiltrationMpi test_task(task_data);
  ASSERT_TRUE(test_task.ValidationImpl());
  ASSERT_TRUE(test_task.PreProcessingImpl());
  ASSERT_TRUE(test_task.RunImpl());
  ASSERT_TRUE(test_task.PostProcessingImpl());

  if (world.rank() == 0) {
    for (unsigned int i = 0; i < received_image.size(); i++) {
      ASSERT_EQ(received_image[i], expected_image[i]) << "Difference at i=" << i;
    }
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
    image = sharamygina_i_horizontal_line_filtration_mpi::GetImage(rows, cols);
    expected_image = sharamygina_i_horizontal_line_filtration_mpi::ToFiltSeq(image, rows, cols);

    task_data->inputs_count.emplace_back(rows);
    task_data->inputs_count.emplace_back(cols);

    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(image.data()));

    received_image.resize(rows * cols);
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(received_image.data()));
    task_data->outputs_count.emplace_back(received_image.size());
  }

  sharamygina_i_horizontal_line_filtration_mpi::HorizontalLineFiltrationMpi test_task(task_data);
  ASSERT_TRUE(test_task.ValidationImpl());
  ASSERT_TRUE(test_task.PreProcessingImpl());
  ASSERT_TRUE(test_task.RunImpl());
  ASSERT_TRUE(test_task.PostProcessingImpl());

  if (world.rank() == 0) {
    for (unsigned int i = 0; i < received_image.size(); i++) {
      ASSERT_EQ(received_image[i], expected_image[i]) << "Difference at i=" << i;
    }
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
    image = sharamygina_i_horizontal_line_filtration_mpi::GetImage(rows, cols);
    expected_image = sharamygina_i_horizontal_line_filtration_mpi::ToFiltSeq(image, rows, cols);

    task_data->inputs_count.emplace_back(rows);
    task_data->inputs_count.emplace_back(cols);
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(image.data()));

    received_image.resize(rows * cols);
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(received_image.data()));
    task_data->outputs_count.emplace_back(received_image.size());
  }

  sharamygina_i_horizontal_line_filtration_mpi::HorizontalLineFiltrationMpi test_task(task_data);
  ASSERT_TRUE(test_task.ValidationImpl());
  ASSERT_TRUE(test_task.PreProcessingImpl());
  ASSERT_TRUE(test_task.RunImpl());
  ASSERT_TRUE(test_task.PostProcessingImpl());

  if (world.rank() == 0) {
    for (unsigned int i = 0; i < received_image.size(); i++) {
      ASSERT_EQ(received_image[i], expected_image[i]) << "Difference at i=" << i;
    }
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
    image = sharamygina_i_horizontal_line_filtration_mpi::GetImage(rows, cols);
    expected_image = sharamygina_i_horizontal_line_filtration_mpi::ToFiltSeq(image, rows, cols);

    task_data->inputs_count.emplace_back(rows);
    task_data->inputs_count.emplace_back(cols);

    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(image.data()));

    received_image.resize(rows * cols);
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(received_image.data()));
    task_data->outputs_count.emplace_back(received_image.size());
  }

  sharamygina_i_horizontal_line_filtration_mpi::HorizontalLineFiltrationMpi test_task(task_data);
  ASSERT_TRUE(test_task.ValidationImpl());
  ASSERT_TRUE(test_task.PreProcessingImpl());
  ASSERT_TRUE(test_task.RunImpl());
  ASSERT_TRUE(test_task.PostProcessingImpl());

  if (world.rank() == 0) {
    for (unsigned int i = 0; i < received_image.size(); i++) {
      ASSERT_EQ(received_image[i], expected_image[i]) << "Difference at i=" << i;
    }
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
    image = sharamygina_i_horizontal_line_filtration_mpi::GetImage(rows, cols);
    expected_image = sharamygina_i_horizontal_line_filtration_mpi::ToFiltSeq(image, rows, cols);

    task_data->inputs_count.emplace_back(rows);
    task_data->inputs_count.emplace_back(cols);

    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(image.data()));

    received_image.resize(rows * cols);
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(received_image.data()));
    task_data->outputs_count.emplace_back(received_image.size());
  }

  sharamygina_i_horizontal_line_filtration_mpi::HorizontalLineFiltrationMpi test_task(task_data);
  ASSERT_TRUE(test_task.ValidationImpl());
  ASSERT_TRUE(test_task.PreProcessingImpl());
  ASSERT_TRUE(test_task.RunImpl());
  ASSERT_TRUE(test_task.PostProcessingImpl());

  if (world.rank() == 0) {
    for (unsigned int i = 0; i < received_image.size(); i++) {
      ASSERT_EQ(received_image[i], expected_image[i]) << "Difference at i=" << i;
    }
  }
}