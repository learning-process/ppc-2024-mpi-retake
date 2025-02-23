#include <gtest/gtest.h>
#include <mpi.h>

#include <boost/mpi.hpp>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "core/task/include/task.hpp"
#include "core/util/include/util.hpp"
#include "mpi/karaseva_e_binaryimage/include/ops_mpi.hpp"

// Function to generate a random binary image
std::vector<int> getRandomBinImage(int r, int c) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vec(r * c);
  for (int i = 0; i < r * c; i++) {
    vec[i] = gen() % 2;
  }
  return vec;
}

TEST(karaseva_e_binaryimage_mpi, test_on_random_25x25) {
  boost::mpi::communicator world;
  const int rows = 25;
  const int cols = 25;

  std::vector<int> image;
  std::vector<int> global_labeled_image(rows * cols);

  // Create TaskData for parallel execution
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    image = getRandomBinImage(rows, cols);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(image.data()));
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->inputs_count.emplace_back(cols);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_labeled_image.data()));
    taskDataPar->outputs_count.emplace_back(rows);
    taskDataPar->outputs_count.emplace_back(cols);
  }

  // Parallel execution test
  karaseva_e_binaryimage_mpi::TestTaskMPI testTaskMPI(taskDataPar);
  ASSERT_TRUE(testTaskMPI.Validation());
  testTaskMPI.PreProcessingImpl();
  testTaskMPI.RunImpl();
  testTaskMPI.PostProcessingImpl();

  if (world.rank() == 0) {
    // Result for sequential processing
    std::vector<int> reference_labeled_image(rows * cols);

    // Create TaskData for sequential execution
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(image.data()));
    taskDataSeq->inputs_count.emplace_back(rows);
    taskDataSeq->inputs_count.emplace_back(cols);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_labeled_image.data()));
    taskDataSeq->outputs_count.emplace_back(rows);
    taskDataSeq->outputs_count.emplace_back(cols);

    // Sequential execution test
    karaseva_e_binaryimage_mpi::TestTaskMPI testTaskSeq(taskDataSeq);
    ASSERT_TRUE(testTaskSeq.Validation());
    testTaskSeq.PreProcessingImpl();
    testTaskSeq.RunImpl();
    testTaskSeq.PostProcessingImpl();

    // Check that the results of parallel and sequential execution match
    ASSERT_EQ(reference_labeled_image, global_labeled_image);
  }
}

TEST(karaseva_e_binaryimage_mpi, test_chessboard_10x10) {
  boost::mpi::communicator world;
  const int rows = 10;
  const int cols = 10;

  // Fixed binary image (e.g., a "checkerboard" pattern)
  std::vector<int> image = {0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0,
                            1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
                            1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1,
                            0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0};
  std::vector<int> global_labeled_image(rows * cols);

  // Create TaskData for parallel execution
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(image.data()));
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->inputs_count.emplace_back(cols);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_labeled_image.data()));
    taskDataPar->outputs_count.emplace_back(rows);
    taskDataPar->outputs_count.emplace_back(cols);
  }

  // Parallel execution test
  karaseva_e_binaryimage_mpi::TestTaskMPI testTaskMPI(taskDataPar);
  ASSERT_TRUE(testTaskMPI.Validation());
  testTaskMPI.PreProcessingImpl();
  testTaskMPI.RunImpl();
  testTaskMPI.PostProcessingImpl();

  if (world.rank() == 0) {
    // Result for sequential processing
    std::vector<int> reference_labeled_image(rows * cols);

    // Create TaskData for sequential execution
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(image.data()));
    taskDataSeq->inputs_count.emplace_back(rows);
    taskDataSeq->inputs_count.emplace_back(cols);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_labeled_image.data()));
    taskDataSeq->outputs_count.emplace_back(rows);
    taskDataSeq->outputs_count.emplace_back(cols);

    // Sequential execution test
    karaseva_e_binaryimage_mpi::TestTaskMPI testTaskSeq(taskDataSeq);
    ASSERT_TRUE(testTaskSeq.Validation());
    testTaskSeq.PreProcessingImpl();
    testTaskSeq.RunImpl();
    testTaskSeq.PostProcessingImpl();

    // Check that the results of parallel and sequential execution match
    ASSERT_EQ(reference_labeled_image, global_labeled_image);
  }
}

TEST(karaseva_e_binaryimage_mpi, test_on_random_50x50) {
  boost::mpi::communicator world;
  const int rows = 50;
  const int cols = 50;

  std::vector<int> image;
  std::vector<int> global_labeled_image(rows * cols);

  // Create TaskData for parallel execution
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    image = getRandomBinImage(rows, cols);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(image.data()));
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->inputs_count.emplace_back(cols);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_labeled_image.data()));
    taskDataPar->outputs_count.emplace_back(rows);
    taskDataPar->outputs_count.emplace_back(cols);
  }

  // Parallel execution test
  karaseva_e_binaryimage_mpi::TestTaskMPI testTaskMPI(taskDataPar);
  ASSERT_TRUE(testTaskMPI.Validation());
  testTaskMPI.PreProcessingImpl();
  testTaskMPI.RunImpl();
  testTaskMPI.PostProcessingImpl();

  if (world.rank() == 0) {
    // Result for sequential processing
    std::vector<int> reference_labeled_image(rows * cols);

    // Create TaskData for sequential execution
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(image.data()));
    taskDataSeq->inputs_count.emplace_back(rows);
    taskDataSeq->inputs_count.emplace_back(cols);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_labeled_image.data()));
    taskDataSeq->outputs_count.emplace_back(rows);
    taskDataSeq->outputs_count.emplace_back(cols);

    // Sequential execution test
    karaseva_e_binaryimage_mpi::TestTaskMPI testTaskSeq(taskDataSeq);
    ASSERT_TRUE(testTaskSeq.Validation());
    testTaskSeq.PreProcessingImpl();
    testTaskSeq.RunImpl();
    testTaskSeq.PostProcessingImpl();

    // Check that the results of parallel and sequential execution match
    ASSERT_EQ(reference_labeled_image, global_labeled_image);
  }
}

TEST(karaseva_e_binaryimage_mpi, test_component_labeling_random) {
  boost::mpi::communicator world;
  constexpr size_t kCount = 10;

  // Generate a random binary image of size 10x10
  std::vector<int> in = {
      0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0,
      0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0,
      0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0,
  };
  std::vector<int> out(kCount * kCount, 0);

  // Create TaskData for parallel execution
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskDataPar->inputs_count.emplace_back(in.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  // Parallel execution test
  karaseva_e_binaryimage_mpi::TestTaskMPI testTaskMPI(taskDataPar);
  ASSERT_TRUE(testTaskMPI.Validation());
  testTaskMPI.PreProcessingImpl();
  testTaskMPI.RunImpl();
  testTaskMPI.PostProcessingImpl();

  if (world.rank() == 0) {
    // Result for sequential processing
    std::vector<int> reference_labeled_image(kCount * kCount);

    // Create TaskData for sequential execution
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskDataSeq->inputs_count.emplace_back(kCount);
    taskDataSeq->inputs_count.emplace_back(kCount);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_labeled_image.data()));
    taskDataSeq->outputs_count.emplace_back(kCount);
    taskDataSeq->outputs_count.emplace_back(kCount);

    // Sequential execution test
    karaseva_e_binaryimage_mpi::TestTaskMPI testTaskSeq(taskDataSeq);
    ASSERT_TRUE(testTaskSeq.Validation());
    testTaskSeq.PreProcessingImpl();
    testTaskSeq.RunImpl();
    testTaskSeq.PostProcessingImpl();

    // Check that the results of parallel and sequential execution match
    ASSERT_EQ(reference_labeled_image, out);
  }
}