#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "mpi/leontev_n_binary/include/ops_mpi.hpp"

namespace {
inline void TaskEmplacement(std::shared_ptr<ppc::core::TaskData>& task_data_par, std::vector<uint8_t>& input,
                            size_t rows, size_t cols, std::vector<uint32_t>& output) {
  task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  task_data_par->inputs_count.emplace_back(rows);
  task_data_par->inputs_count.emplace_back(cols);
  task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
  task_data_par->outputs_count.emplace_back(rows);
  task_data_par->outputs_count.emplace_back(cols);
}
}  // namespace

TEST(leontev_n_binary_mpi, input_1) {
  boost::mpi::communicator world;
  size_t rows = 4;
  size_t cols = 4;
  std::vector<uint8_t> img = {0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0};
  std::vector<uint32_t> expected = {0, 1, 0, 2, 1, 1, 0, 2, 0, 0, 0, 2, 3, 0, 4, 0};
  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();
  std::vector<uint32_t> actual(rows * cols);
  if (world.rank() == 0) {
    TaskEmplacement(task_data_par, img, rows, cols, actual);
  }
  leontev_n_binary_mpi::BinarySegmentsMPI binary_segments(task_data_par);
  binary_segments.Validation();
  binary_segments.PreProcessing();
  binary_segments.Run();
  binary_segments.PostProcessing();
  if (world.rank() == 0) {
    ASSERT_EQ(actual, expected);
  }
}

TEST(leontev_n_binary_mpi, input_2) {
  boost::mpi::communicator world;
  size_t rows = 4;
  size_t cols = 4;
  std::vector<uint8_t> img = {0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1};
  std::vector<uint32_t> expected = {0, 1, 1, 0, 2, 0, 0, 1, 0, 2, 0, 0, 0, 0, 0, 3};
  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();
  std::vector<uint32_t> actual(rows * cols);
  if (world.rank() == 0) {
    TaskEmplacement(task_data_par, img, rows, cols, actual);
  }
  leontev_n_binary_mpi::BinarySegmentsMPI binary_segments(task_data_par);
  binary_segments.Validation();
  binary_segments.PreProcessing();
  binary_segments.Run();
  binary_segments.PostProcessing();
  if (world.rank() == 0) {
    ASSERT_EQ(actual, expected);
  }
}

TEST(leontev_n_binary_mpi, input_3) {
  boost::mpi::communicator world;
  size_t rows = 1;
  size_t cols = 5;
  std::vector<uint8_t> img = {0, 1, 1, 0, 1};
  std::vector<uint32_t> expected = {0, 1, 1, 0, 2};
  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();
  std::vector<uint32_t> actual(rows * cols);
  if (world.rank() == 0) {
    TaskEmplacement(task_data_par, img, rows, cols, actual);
  }
  leontev_n_binary_mpi::BinarySegmentsMPI binary_segments(task_data_par);
  binary_segments.Validation();
  binary_segments.PreProcessing();
  binary_segments.Run();
  binary_segments.PostProcessing();
  if (world.rank() == 0) {
    ASSERT_EQ(actual, expected);
  }
}

TEST(leontev_n_binary_mpi, input_4) {
  boost::mpi::communicator world;
  size_t rows = 5;
  size_t cols = 1;
  std::vector<uint8_t> img = {0, 1, 1, 0, 1};
  std::vector<uint32_t> expected = {0, 1, 1, 0, 2};
  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();
  std::vector<uint32_t> actual(rows * cols);
  if (world.rank() == 0) {
    TaskEmplacement(task_data_par, img, rows, cols, actual);
  }
  leontev_n_binary_mpi::BinarySegmentsMPI binary_segments(task_data_par);
  binary_segments.Validation();
  binary_segments.PreProcessing();
  binary_segments.Run();
  binary_segments.PostProcessing();
  if (world.rank() == 0) {
    ASSERT_EQ(actual, expected);
  }
}

TEST(leontev_n_binary_mpi, input_5) {
  boost::mpi::communicator world;
  size_t rows = 7;
  size_t cols = 7;
  std::vector<uint8_t> img = {0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0,
                              0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0};
  std::vector<uint32_t> expected = {0, 1, 0, 0, 0, 2, 0, 3, 0, 1, 0, 4, 0, 2, 0, 3, 0, 0, 0, 4, 0, 0, 3, 0, 0,
                                    0, 0, 0, 5, 0, 3, 0, 0, 6, 0, 0, 5, 0, 0, 7, 0, 6, 0, 0, 0, 0, 0, 7, 0};
  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();
  std::vector<uint32_t> actual(rows * cols);
  if (world.rank() == 0) {
    TaskEmplacement(task_data_par, img, rows, cols, actual);
  }
  leontev_n_binary_mpi::BinarySegmentsMPI binary_segments(task_data_par);
  binary_segments.Validation();
  binary_segments.PreProcessing();
  binary_segments.Run();
  binary_segments.PostProcessing();
  if (world.rank() == 0) {
    ASSERT_EQ(actual, expected);
  }
}

TEST(leontev_n_binary_mpi, empty_test) {
  boost::mpi::communicator world;
  size_t rows = 7;
  size_t cols = 7;
  std::vector<uint8_t> img = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<uint32_t> expected = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();
  std::vector<uint32_t> actual(rows * cols);
  if (world.rank() == 0) {
    TaskEmplacement(task_data_par, img, rows, cols, actual);
  }
  leontev_n_binary_mpi::BinarySegmentsMPI binary_segments(task_data_par);
  binary_segments.Validation();
  binary_segments.PreProcessing();
  binary_segments.Run();
  binary_segments.PostProcessing();
  if (world.rank() == 0) {
    ASSERT_EQ(actual, expected);
  }
}

TEST(leontev_n_binary_mpi, lines_test) {
  boost::mpi::communicator world;
  size_t rows = 7;
  size_t cols = 7;
  std::vector<uint8_t> img = {1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
                              0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1};
  std::vector<uint32_t> expected = {1, 0, 2, 0, 3, 0, 4, 0, 1, 0, 2, 0, 3, 0, 5, 0, 1, 0, 2, 0, 3, 0, 5, 0, 1,
                                    0, 2, 0, 6, 0, 5, 0, 1, 0, 2, 0, 6, 0, 5, 0, 1, 0, 7, 0, 6, 0, 5, 0, 1};
  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();
  std::vector<uint32_t> actual(rows * cols);
  if (world.rank() == 0) {
    TaskEmplacement(task_data_par, img, rows, cols, actual);
  }
  leontev_n_binary_mpi::BinarySegmentsMPI binary_segments(task_data_par);
  binary_segments.Validation();
  binary_segments.PreProcessing();
  binary_segments.Run();
  binary_segments.PostProcessing();
  if (world.rank() == 0) {
    ASSERT_EQ(actual, expected);
  }
}
