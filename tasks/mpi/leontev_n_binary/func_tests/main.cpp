#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <unordered_map>
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

std::vector<uint8_t> GetRandomVector(size_t rows, size_t cols) {
  std::vector<uint8_t> img(rows * cols);
  for (size_t i = 0; i < img.size(); i++) {
    img[i] = rand() % 2;
  }
  return img;
}

bool CompNotZero(uint32_t a, uint32_t b) {
  if (a == 0) {
    return false;
  }
  if (b == 0) {
    return true;
  }
  return a < b;
}

std::vector<uint32_t> RunSeq(const std::vector<uint8_t>& image, size_t rows, size_t cols) {
  std::vector<uint32_t> labels(rows * cols, 0);
  std::unordered_map<uint32_t, uint32_t> label_equivalences;
  uint32_t cur_label = 1;
  for (size_t row = 0; row < rows; ++row) {
    for (size_t col = 0; col < cols; ++col) {
      if (image[(row * cols) + col] == 0) {
        continue;
      }
      uint32_t label_b = (col > 0) ? labels[(row * cols) + col - 1] : 0;
      uint32_t label_c = (row > 0) ? labels[((row - 1) * cols) + col] : 0;
      uint32_t label_d = (row > 0 && col > 0) ? labels[((row - 1) * cols) + col - 1] : 0;

      if (label_b == 0 && label_c == 0 && label_d == 0) {
        labels[(row * cols) + col] = cur_label++;
      } else {
        uint32_t min_label = std::min({label_b, label_c, label_d}, CompNotZero);
        labels[(row * cols) + col] = min_label;
        for (uint32_t label : {label_b, label_c, label_d}) {
          if (label != 0 && label != min_label) {
            label_equivalences[std::max(label, min_label)] = std::min(label, min_label);
          }
        }
      }
    }
  }
  for (auto& label : labels) {
    while (label_equivalences.contains(label)) {
      label = label_equivalences[label];
    }
  }
  std::vector<size_t> arrived((rows * cols) + 1, 0);
  size_t cur_mark = 1;
  for (size_t i = 0; i < rows * cols; i++) {
    if (labels[i] != 0) {
      if (arrived[labels[i]] != 0) {
        labels[i] = arrived[labels[i]];
      } else {
        labels[i] = arrived[labels[i]] = cur_mark++;
      }
    }
  }
  return labels;
}
}  // namespace

TEST(leontev_n_binary_mpi, random_test) {
  boost::mpi::communicator world;
  size_t rows = 10;
  size_t cols = 10;
  std::vector<uint8_t> img = GetRandomVector(rows, cols);
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
    std::vector<uint32_t> expected = RunSeq(img, rows, cols);
    ASSERT_EQ(actual, expected);
  }
}

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
