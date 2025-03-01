#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/leontev_n_binary/include/ops_seq.hpp"

namespace {
inline void TaskEmplacement(std::shared_ptr<ppc::core::TaskData>& task_data_seq, std::vector<uint8_t>& input,
                            size_t rows, size_t cols, std::vector<uint32_t>& output) {
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  task_data_seq->inputs_count.emplace_back(rows);
  task_data_seq->inputs_count.emplace_back(cols);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
  task_data_seq->outputs_count.emplace_back(rows);
  task_data_seq->outputs_count.emplace_back(cols);
}
}  // namespace

TEST(leontev_n_binary_seq, full_square_test) {
  std::vector<uint8_t> img = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

  std::vector<uint32_t> expected = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  size_t rows = 4;
  size_t cols = 4;
  std::vector<uint32_t> actual(expected.size());
  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
  TaskEmplacement(task_data_seq, img, rows, cols, actual);
  leontev_n_binary_seq::BinarySegmentsSeq binary_segments(task_data_seq);
  ASSERT_TRUE(binary_segments.Validation());
  binary_segments.PreProcessing();
  binary_segments.Run();
  binary_segments.PostProcessing();
  ASSERT_EQ(actual, expected);
}

TEST(leontev_n_binary_seq, cross_test) {
  std::vector<uint8_t> img = {1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1};

  std::vector<uint32_t> expected = {1, 0, 0, 2, 0, 1, 1, 0, 0, 1, 1, 0, 3, 0, 0, 1};
  size_t rows = 4;
  size_t cols = 4;
  std::vector<uint32_t> actual(expected.size());
  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
  TaskEmplacement(task_data_seq, img, rows, cols, actual);
  leontev_n_binary_seq::BinarySegmentsSeq binary_segments(task_data_seq);
  ASSERT_TRUE(binary_segments.Validation());
  binary_segments.PreProcessing();
  binary_segments.Run();
  binary_segments.PostProcessing();
  ASSERT_EQ(actual, expected);
}

TEST(leontev_n_binary_seq, circles_test) {
  std::vector<uint8_t> img = {0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0,
                              0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0};

  std::vector<uint32_t> expected = {0, 1, 0, 0, 0, 2, 0, 3, 0, 1, 0, 4, 0, 2, 0, 3, 0, 0, 0, 4, 0, 0, 3, 0, 0,
                                    0, 0, 0, 5, 0, 3, 0, 0, 6, 0, 0, 5, 0, 0, 7, 0, 6, 0, 0, 0, 0, 0, 7, 0};
  size_t rows = 7;
  size_t cols = 7;
  std::vector<uint32_t> actual(expected.size());
  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
  TaskEmplacement(task_data_seq, img, rows, cols, actual);
  leontev_n_binary_seq::BinarySegmentsSeq binary_segments(task_data_seq);
  ASSERT_TRUE(binary_segments.Validation());
  binary_segments.PreProcessing();
  binary_segments.Run();
  binary_segments.PostProcessing();
  ASSERT_EQ(actual, expected);
}

TEST(leontev_n_binary_seq, empty_test) {
  std::vector<uint8_t> img = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  std::vector<uint32_t> expected = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  size_t rows = 7;
  size_t cols = 7;
  std::vector<uint32_t> actual(expected.size());
  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
  TaskEmplacement(task_data_seq, img, rows, cols, actual);
  leontev_n_binary_seq::BinarySegmentsSeq binary_segments(task_data_seq);
  ASSERT_TRUE(binary_segments.Validation());
  binary_segments.PreProcessing();
  binary_segments.Run();
  binary_segments.PostProcessing();
  ASSERT_EQ(actual, expected);
}

TEST(leontev_n_binary_seq, lines_test) {
  std::vector<uint8_t> img = {1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
                              0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1};

  std::vector<uint32_t> expected = {1, 0, 2, 0, 3, 0, 4, 0, 1, 0, 2, 0, 3, 0, 5, 0, 1, 0, 2, 0, 3, 0, 5, 0, 1,
                                    0, 2, 0, 6, 0, 5, 0, 1, 0, 2, 0, 6, 0, 5, 0, 1, 0, 7, 0, 6, 0, 5, 0, 1};
  size_t rows = 7;
  size_t cols = 7;
  std::vector<uint32_t> actual(expected.size());
  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
  TaskEmplacement(task_data_seq, img, rows, cols, actual);
  leontev_n_binary_seq::BinarySegmentsSeq binary_segments(task_data_seq);
  ASSERT_TRUE(binary_segments.Validation());
  binary_segments.PreProcessing();
  binary_segments.Run();
  binary_segments.PostProcessing();
  ASSERT_EQ(actual, expected);
}
