#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include <boost/mpi/communicator.hpp>
#include "core/task/include/task.hpp"
#include "mpi/karaseva_e_binaryimage/include/ops_mpi.hpp"

// Move the function to an anonymous namespace to avoid the static warning
namespace {
std::vector<int> GetRandomBinImage(int r, int c) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_int_distribution<int> dist(0, 1);
  std::vector<int> vec(r * c);
  for (int i = 0; i < r * c; i++) {
    vec[i] = dist(gen);
  }
  return vec;
}
}  // namespace

TEST(karaseva_e_binaryimage_mpi, test_on_random_25x25) {
  boost::mpi::communicator world;
  const int rows = 25;
  const int cols = 25;

  std::vector<int> image;
  std::vector<int> global_labeled_image(rows * cols);

  // Create TaskData for parallel execution
  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    image = GetRandomBinImage(rows, cols);
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(image.data()));
    task_data_par->inputs_count.emplace_back(rows);
    task_data_par->inputs_count.emplace_back(cols);
    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_labeled_image.data()));
    task_data_par->outputs_count.emplace_back(rows);
    task_data_par->outputs_count.emplace_back(cols);
  }

  // Parallel execution test
  karaseva_e_binaryimage_mpi::TestTaskMPI test_task_mpi(task_data_par);
  ASSERT_TRUE(test_task_mpi.Validation());
  test_task_mpi.PreProcessingImpl();
  test_task_mpi.RunImpl();
  test_task_mpi.PostProcessingImpl();

  if (world.rank() == 0) {
    // Result for sequential processing
    std::vector<int> reference_labeled_image(rows * cols);

    // Create TaskData for sequential execution
    std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(image.data()));
    task_data_seq->inputs_count.emplace_back(rows);
    task_data_seq->inputs_count.emplace_back(cols);
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_labeled_image.data()));
    task_data_seq->outputs_count.emplace_back(rows);
    task_data_seq->outputs_count.emplace_back(cols);

    // Sequential execution test
    karaseva_e_binaryimage_mpi::TestTaskMPI test_task_seq(task_data_seq);
    ASSERT_TRUE(test_task_seq.Validation());
    test_task_seq.PreProcessingImpl();
    test_task_seq.RunImpl();
    test_task_seq.PostProcessingImpl();

    // Check that the results of parallel and sequential execution match
    ASSERT_EQ(reference_labeled_image, global_labeled_image);
  }
}

TEST(karaseva_e_binaryimage_mpi, test_chessboard_10x10) {
  boost::mpi::communicator world;
  const int rows = 10;
  const int cols = 10;

  // Fixed binary image
  std::vector<int> image = {0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0,
                            1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
                            1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1,
                            0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0};
  std::vector<int> global_labeled_image(rows * cols);

  // Create TaskData for parallel execution
  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(image.data()));
    task_data_par->inputs_count.emplace_back(rows);
    task_data_par->inputs_count.emplace_back(cols);
    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_labeled_image.data()));
    task_data_par->outputs_count.emplace_back(rows);
    task_data_par->outputs_count.emplace_back(cols);
  }

  // Parallel execution test
  karaseva_e_binaryimage_mpi::TestTaskMPI test_task_mpi(task_data_par);
  ASSERT_TRUE(test_task_mpi.Validation());
  test_task_mpi.PreProcessingImpl();
  test_task_mpi.RunImpl();
  test_task_mpi.PostProcessingImpl();

  if (world.rank() == 0) {
    // Result for sequential processing
    std::vector<int> reference_labeled_image(rows * cols);

    // Create TaskData for sequential execution
    std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(image.data()));
    task_data_seq->inputs_count.emplace_back(rows);
    task_data_seq->inputs_count.emplace_back(cols);
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_labeled_image.data()));
    task_data_seq->outputs_count.emplace_back(rows);
    task_data_seq->outputs_count.emplace_back(cols);

    // Sequential execution test
    karaseva_e_binaryimage_mpi::TestTaskMPI test_task_seq(task_data_seq);
    ASSERT_TRUE(test_task_seq.Validation());
    test_task_seq.PreProcessingImpl();
    test_task_seq.RunImpl();
    test_task_seq.PostProcessingImpl();

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
  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    image = GetRandomBinImage(rows, cols);
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(image.data()));
    task_data_par->inputs_count.emplace_back(rows);
    task_data_par->inputs_count.emplace_back(cols);
    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_labeled_image.data()));
    task_data_par->outputs_count.emplace_back(rows);
    task_data_par->outputs_count.emplace_back(cols);
  }

  // Parallel execution test
  karaseva_e_binaryimage_mpi::TestTaskMPI test_task_mpi(task_data_par);
  ASSERT_TRUE(test_task_mpi.Validation());
  test_task_mpi.PreProcessingImpl();
  test_task_mpi.RunImpl();
  test_task_mpi.PostProcessingImpl();

  if (world.rank() == 0) {
    // Result for sequential processing
    std::vector<int> reference_labeled_image(rows * cols);

    // Create TaskData for sequential execution
    std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(image.data()));
    task_data_seq->inputs_count.emplace_back(rows);
    task_data_seq->inputs_count.emplace_back(cols);
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_labeled_image.data()));
    task_data_seq->outputs_count.emplace_back(rows);
    task_data_seq->outputs_count.emplace_back(cols);

    // Sequential execution test
    karaseva_e_binaryimage_mpi::TestTaskMPI test_task_seq(task_data_seq);
    ASSERT_TRUE(test_task_seq.Validation());
    test_task_seq.PreProcessingImpl();
    test_task_seq.RunImpl();
    test_task_seq.PostProcessingImpl();

    // Check that the results of parallel and sequential execution match
    ASSERT_EQ(reference_labeled_image, global_labeled_image);
  }
}