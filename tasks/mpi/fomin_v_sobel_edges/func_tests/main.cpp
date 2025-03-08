#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/fomin_v_sobel_edges/include/ops_mpi.hpp"

TEST(fomin_v_sobel_edges, Test_Sobel_Edge_Detection) {
  boost::mpi::communicator world;
  std::vector<unsigned char> global_image;
  std::vector<unsigned char> global_output_image;

  const int width = 4;
  const int height = 4;

  if (world.size() > 4) {
    GTEST_SKIP() << "Skipping for large process count";
  }

  if (world.rank() == 0) {
    global_image = {100, 100, 100, 100, 100, 200, 200, 100, 100, 200, 200, 100, 100, 100, 100, 100};
    global_output_image.resize(width * height, 0);
  }

  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();

  task_data->inputs_count = {width, height};
  task_data->outputs_count = {width, height};

  if (world.rank() == 0) {
    if (!global_image.data() || !global_output_image.data()) {
      std::cerr << "Error: Test buffers not initialized" << std::endl;
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
    task_data->inputs.push_back(global_image.data());
    task_data->outputs.push_back(global_output_image.data());
  } else {
    task_data->inputs.push_back(nullptr);
    task_data->outputs.push_back(nullptr);
  }

  fomin_v_sobel_edges::SobelEdgeDetectionMPI sobelEdgeDetectionMPI(task_data);
  ASSERT_EQ(sobelEdgeDetectionMPI.ValidationImpl(), true);
  sobelEdgeDetectionMPI.PreProcessingImpl();
  sobelEdgeDetectionMPI.RunImpl();
  sobelEdgeDetectionMPI.PostProcessingImpl();

  if (world.rank() == 0) {
    // Создание TaskData для последовательной версии
    std::vector<unsigned char> reference_output_image(width * height, 0);
    std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_image.data()));
    task_data_seq->inputs_count.emplace_back(width);
    task_data_seq->inputs_count.emplace_back(height);
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_output_image.data()));
    task_data_seq->outputs_count.emplace_back(width);
    task_data_seq->outputs_count.emplace_back(height);

    // Создание и выполнение последовательной задачи
    fomin_v_sobel_edges::SobelEdgeDetection sobelEdgeDetection(task_data_seq);
    ASSERT_EQ(sobelEdgeDetection.ValidationImpl(), true);
    sobelEdgeDetection.PreProcessingImpl();
    sobelEdgeDetection.RunImpl();
    sobelEdgeDetection.PostProcessingImpl();

    // Сравнение результатов
    for (size_t i = 0; i < reference_output_image.size(); ++i) {
      ASSERT_EQ(reference_output_image[i], global_output_image[i]);
    }
  }
}

TEST(fomin_v_sobel_edges, Test_Sobel_Edge_Detection_Large_Image) {
  boost::mpi::communicator world;
  std::vector<unsigned char> global_image;
  std::vector<unsigned char> global_output_image;

  // Создание большого тестового изображения
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
  fomin_v_sobel_edges::SobelEdgeDetectionMPI sobelEdgeDetectionMPI(task_data);
  ASSERT_EQ(sobelEdgeDetectionMPI.ValidationImpl(), true);
  sobelEdgeDetectionMPI.PreProcessingImpl();
  sobelEdgeDetectionMPI.RunImpl();
  sobelEdgeDetectionMPI.PostProcessingImpl();

  if (world.rank() == 0) {
    // Создание TaskData для последовательной версии
    std::vector<unsigned char> reference_output_image(width * height, 0);
    std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_image.data()));
    task_data_seq->inputs_count.emplace_back(width);
    task_data_seq->inputs_count.emplace_back(height);
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_output_image.data()));
    task_data_seq->outputs_count.emplace_back(width);
    task_data_seq->outputs_count.emplace_back(height);

    // Создание и выполнение последовательной задачи
    fomin_v_sobel_edges::SobelEdgeDetection sobelEdgeDetection(task_data_seq);
    ASSERT_EQ(sobelEdgeDetection.ValidationImpl(), true);
    sobelEdgeDetection.PreProcessingImpl();
    sobelEdgeDetection.RunImpl();
    sobelEdgeDetection.PostProcessingImpl();

    // Сравнение результатов
    for (size_t i = 0; i < reference_output_image.size(); ++i) {
      ASSERT_EQ(reference_output_image[i], global_output_image[i]);
    }
  }
}

TEST(fomin_v_sobel_edges, Test_Sobel_Edge_Detection_Empty_Image) {
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();

  task_data->inputs_count = {0, 0};
  task_data->outputs_count = {0, 0};

  if (world.rank() == 0) {
    task_data->inputs.push_back(nullptr);
    task_data->outputs.push_back(nullptr);
  }

  fomin_v_sobel_edges::SobelEdgeDetectionMPI sobelEdgeDetectionMPI(task_data);
  ASSERT_FALSE(sobelEdgeDetectionMPI.ValidationImpl());
}