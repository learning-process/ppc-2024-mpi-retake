#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "mpi/karaseva_e_binaryimage/include/ops_mpi.hpp"

std::vector<int> createRandomBinaryImage(int r, int c) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vec(r * c);
  for (int i = 0; i < r * c; i++) {
    vec[i] = gen() % 2;
  }
  return vec;
}

TEST(karaseva_e_binaryimage_mpi, test_on_random_ing_25x25) {
  boost::mpi::communicator world;
  const int rows = 25;
  const int cols = 25;
  std::vector<int> image;
  std::vector<int> global_labeled_image(rows * cols);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    image = createRandomBinaryImage(rows, cols);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(image.data()));
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->inputs_count.emplace_back(cols);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_labeled_image.data()));
    taskDataPar->outputs_count.emplace_back(rows);
    taskDataPar->outputs_count.emplace_back(cols);
  }

  karaseva_e_binaryimage_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_TRUE(testMpiTaskParallel.ValidationImpl());
  testMpiTaskParallel.PreProcessingImpl();
  testMpiTaskParallel.RunImpl();
  testMpiTaskParallel.PostProcessingImpl();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> reference_labeled_image(rows * cols);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(image.data()));
    taskDataSeq->inputs_count.emplace_back(rows);
    taskDataSeq->inputs_count.emplace_back(cols);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_labeled_image.data()));
    taskDataSeq->outputs_count.emplace_back(rows);
    taskDataSeq->outputs_count.emplace_back(cols);

    // Create Task
    karaseva_e_binaryimage_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_TRUE(testMpiTaskSequential.ValidationImpl());
    testMpiTaskSequential.PreProcessingImpl();
    testMpiTaskSequential.RunImpl();
    testMpiTaskSequential.PostProcessingImpl();
    ASSERT_EQ(reference_labeled_image, global_labeled_image);
  }
}

TEST(karaseva_e_binaryimage_mpi, test_on_random_ing_50x50) {
  boost::mpi::communicator world;
  const int rows = 50;
  const int cols = 50;
  std::vector<int> image;
  std::vector<int> global_labeled_image(rows * cols);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    image = createRandomBinaryImage(rows, cols);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(image.data()));
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->inputs_count.emplace_back(cols);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_labeled_image.data()));
    taskDataPar->outputs_count.emplace_back(rows);
    taskDataPar->outputs_count.emplace_back(cols);
  }

  karaseva_e_binaryimage_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_TRUE(testMpiTaskParallel.ValidationImpl());
  testMpiTaskParallel.PreProcessingImpl();
  testMpiTaskParallel.RunImpl();
  testMpiTaskParallel.PostProcessingImpl();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> reference_labeled_image(rows * cols);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(image.data()));
    taskDataSeq->inputs_count.emplace_back(rows);
    taskDataSeq->inputs_count.emplace_back(cols);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_labeled_image.data()));
    taskDataSeq->outputs_count.emplace_back(rows);
    taskDataSeq->outputs_count.emplace_back(cols);

    // Create Task
    karaseva_e_binaryimage_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_TRUE(testMpiTaskSequential.ValidationImpl());
    testMpiTaskSequential.PreProcessingImpl();
    testMpiTaskSequential.RunImpl();
    testMpiTaskSequential.PostProcessingImpl();
    ASSERT_EQ(reference_labeled_image, global_labeled_image);
  }
}

TEST(karaseva_e_binaryimage_mpi, test_on_random_ing_75x75) {
  boost::mpi::communicator world;
  const int rows = 75;
  const int cols = 75;
  std::vector<int> image;
  std::vector<int> global_labeled_image(rows * cols);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    image = createRandomBinaryImage(rows, cols);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(image.data()));
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->inputs_count.emplace_back(cols);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_labeled_image.data()));
    taskDataPar->outputs_count.emplace_back(rows);
    taskDataPar->outputs_count.emplace_back(cols);
  }

  karaseva_e_binaryimage_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_TRUE(testMpiTaskParallel.ValidationImpl());
  testMpiTaskParallel.PreProcessingImpl();
  testMpiTaskParallel.RunImpl();
  testMpiTaskParallel.PostProcessingImpl();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> reference_labeled_image(rows * cols);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(image.data()));
    taskDataSeq->inputs_count.emplace_back(rows);
    taskDataSeq->inputs_count.emplace_back(cols);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_labeled_image.data()));
    taskDataSeq->outputs_count.emplace_back(rows);
    taskDataSeq->outputs_count.emplace_back(cols);

    // Create Task
    karaseva_e_binaryimage_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_TRUE(testMpiTaskSequential.ValidationImpl());
    testMpiTaskSequential.PreProcessingImpl();
    testMpiTaskSequential.RunImpl();
    testMpiTaskSequential.PostProcessingImpl();
    ASSERT_EQ(reference_labeled_image, global_labeled_image);
  }
}
TEST(karaseva_e_binaryimage_mpi, predefined_test_50x50) {
  boost::mpi::communicator world;
  const int rows = 50;
  const int cols = 50;

  // Create a predefined binary image with known objects and background
  std::vector<int> image(rows * cols, 0);
  // Fill objects in the image (1)
  for (int i = 10; i < 20; ++i) {
    for (int j = 10; j < 20; ++j) {
      image[i * cols + j] = 1;
    }
  }
  std::vector<int> global_labeled_image(rows * cols);

  // Create TaskData for parallel processing
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(image.data()));
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->inputs_count.emplace_back(cols);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_labeled_image.data()));
    taskDataPar->outputs_count.emplace_back(rows);
    taskDataPar->outputs_count.emplace_back(cols);
  }

  // Run parallel processing
  karaseva_e_binaryimage_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_TRUE(testMpiTaskParallel.ValidationImpl());
  testMpiTaskParallel.PreProcessingImpl();
  testMpiTaskParallel.RunImpl();
  testMpiTaskParallel.PostProcessingImpl();

  if (world.rank() == 0) {
    // Create reference labeled image with known results
    std::vector<int> reference_labeled_image(rows * cols, 0);
    // Example of marking an object in the center
    for (int i = 10; i < 20; ++i) {
      for (int j = 10; j < 20; ++j) {
        reference_labeled_image[i * cols + j] = 1;  // Same object
      }
    }

    // Create TaskData for sequential processing
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(image.data()));
    taskDataSeq->inputs_count.emplace_back(rows);
    taskDataSeq->inputs_count.emplace_back(cols);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_labeled_image.data()));
    taskDataSeq->outputs_count.emplace_back(rows);
    taskDataSeq->outputs_count.emplace_back(cols);

    // Run sequential processing
    karaseva_e_binaryimage_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_TRUE(testMpiTaskSequential.ValidationImpl());
    testMpiTaskSequential.PreProcessingImpl();
    testMpiTaskSequential.RunImpl();
    testMpiTaskSequential.PostProcessingImpl();
    // Compare the results
    ASSERT_EQ(reference_labeled_image, global_labeled_image);
  }
}