#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <climits>
#include <random>
#include <vector>

#include "mpi/malyshev_v_lent_horizontal/include/ops_mpi.hpp"

namespace malyshev_lent_horizontal {

std::vector<std::vector<int32_t>> generateRandomMatrix(uint32_t rows, uint32_t cols, int32_t min_value,
                                                       int32_t max_value) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<std::vector<int32_t>> data(rows, std::vector<int32_t>(cols));

  for (auto &row : data) {
    for (auto &el : row) {
      el = min_value + gen() % (max_value - min_value + 1);
    }
  }

  return data;
}

std::vector<int32_t> generateRandomVector(uint32_t size, int32_t min_value, int32_t max_value) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int32_t> data(size);

  for (auto &el : data) {
    el = min_value + gen() % (max_value - min_value + 1);
  }

  return data;
}

}  // namespace malyshev_lent_horizontal

TEST(malyshev_lent_horizontal, test_empty_matrix) {
  boost::mpi::communicator world;

  uint32_t rows = 0;
  uint32_t cols = 0;

  // Create data
  std::vector<std::vector<int32_t>> in = {};
  std::vector<int32_t> out_par = {};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_par->inputs_count.emplace_back(in.size());
    task_data_par->inputs_count.emplace_back(rows);
    task_data_par->inputs_count.emplace_back(cols);
    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_par.data()));
    task_data_par->outputs_count.emplace_back(out_par.size());
  }

  malyshev_lent_horizontal::TestTaskParallel taskMPI(task_data_par);
  ASSERT_EQ(taskMPI.ValidationImpl(), true);
  taskMPI.PreProcessingImpl();
  taskMPI.RunImpl();
  taskMPI.PostProcessingImpl();
}

TEST(malyshev_lent_horizontal, test_1x1_matrix) {
  boost::mpi::communicator world;

  uint32_t rows = 1;
  uint32_t cols = 1;
  int32_t min_value = -200;
  int32_t max_value = 300;

  // Create data
  std::vector<std::vector<int32_t>> in =
      malyshev_lent_horizontal::generateRandomMatrix(rows, cols, min_value, max_value);
  std::vector<int32_t> vec = malyshev_lent_horizontal::generateRandomVector(cols, min_value, max_value);
  std::vector<int32_t> out_par(rows, 0);

  std::vector<int32_t> expect(rows, 0);
  for (uint32_t i = 0; i < rows; i++) {
    for (uint32_t j = 0; j < cols; j++) {
      expect[i] += in[i][j] * vec[j];
    }
  }

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    for (auto &row : in) {
      task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(row.data()));
    }
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(vec.data()));
    task_data_par->inputs_count.emplace_back(rows);
    task_data_par->inputs_count.emplace_back(cols);
    task_data_par->inputs_count.emplace_back(cols);
    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_par.data()));
    task_data_par->outputs_count.emplace_back(out_par.size());
  }

  malyshev_lent_horizontal::TestTaskParallel taskMPI(task_data_par);
  ASSERT_EQ(taskMPI.ValidationImpl(), true);
  taskMPI.PreProcessingImpl();
  taskMPI.RunImpl();
  taskMPI.PostProcessingImpl();

  if (world.rank() == 0) {
    ASSERT_EQ(out_par, expect);
  }
}

TEST(malyshev_lent_horizontal, test_1x_matrix) {
  boost::mpi::communicator world;

  uint32_t rows = 1;
  uint32_t cols = 5;
  int32_t min_value = -200;
  int32_t max_value = 300;

  // Create data
  std::vector<std::vector<int32_t>> in =
      malyshev_lent_horizontal::generateRandomMatrix(rows, cols, min_value, max_value);
  std::vector<int32_t> vec = malyshev_lent_horizontal::generateRandomVector(cols, min_value, max_value);
  std::vector<int32_t> out_par(rows, 0);

  std::vector<int32_t> expect(rows, 0);
  for (uint32_t i = 0; i < rows; i++) {
    for (uint32_t j = 0; j < cols; j++) {
      expect[i] += in[i][j] * vec[j];
    }
  }

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    for (auto &row : in) {
      task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(row.data()));
    }
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(vec.data()));
    task_data_par->inputs_count.emplace_back(rows);
    task_data_par->inputs_count.emplace_back(cols);
    task_data_par->inputs_count.emplace_back(cols);
    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_par.data()));
    task_data_par->outputs_count.emplace_back(out_par.size());
  }

  malyshev_lent_horizontal::TestTaskParallel taskMPI(task_data_par);
  ASSERT_EQ(taskMPI.ValidationImpl(), true);
  taskMPI.PreProcessingImpl();
  taskMPI.RunImpl();
  taskMPI.PostProcessingImpl();

  if (world.rank() == 0) {
    ASSERT_EQ(out_par, expect);
  }
}

TEST(malyshev_lent_horizontal, test_x1_matrix) {
  boost::mpi::communicator world;

  uint32_t rows = 5;
  uint32_t cols = 1;
  int32_t min_value = -200;
  int32_t max_value = 300;

  // Create data
  std::vector<std::vector<int32_t>> in =
      malyshev_lent_horizontal::generateRandomMatrix(rows, cols, min_value, max_value);
  std::vector<int32_t> vec = malyshev_lent_horizontal::generateRandomVector(cols, min_value, max_value);
  std::vector<int32_t> out_par(rows, 0);

  std::vector<int32_t> expect(rows, 0);
  for (uint32_t i = 0; i < rows; i++) {
    for (uint32_t j = 0; j < cols; j++) {
      expect[i] += in[i][j] * vec[j];
    }
  }

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    for (auto &row : in) {
      task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(row.data()));
    }
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(vec.data()));
    task_data_par->inputs_count.emplace_back(rows);
    task_data_par->inputs_count.emplace_back(cols);
    task_data_par->inputs_count.emplace_back(cols);
    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_par.data()));
    task_data_par->outputs_count.emplace_back(out_par.size());
  }

  malyshev_lent_horizontal::TestTaskParallel taskMPI(task_data_par);
  ASSERT_EQ(taskMPI.ValidationImpl(), true);
  taskMPI.PreProcessingImpl();
  taskMPI.RunImpl();
  taskMPI.PostProcessingImpl();

  if (world.rank() == 0) {
    ASSERT_EQ(out_par, expect);
  }
}

TEST(malyshev_lent_horizontal, test_random_matrix) {
  boost::mpi::communicator world;

  uint32_t rows = 10;
  uint32_t cols = 10;
  int32_t min_value = -200;
  int32_t max_value = 300;

  // Create data
  std::vector<std::vector<int32_t>> in =
      malyshev_lent_horizontal::generateRandomMatrix(rows, cols, min_value, max_value);
  std::vector<int32_t> vec = malyshev_lent_horizontal::generateRandomVector(cols, min_value, max_value);
  std::vector<int32_t> out_par(rows, 0);

  std::vector<int32_t> expect(rows, 0);
  for (uint32_t i = 0; i < rows; i++) {
    for (uint32_t j = 0; j < cols; j++) {
      expect[i] += in[i][j] * vec[j];
    }
  }

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    for (auto &row : in) {
      task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(row.data()));
    }
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(vec.data()));
    task_data_par->inputs_count.emplace_back(rows);
    task_data_par->inputs_count.emplace_back(cols);
    task_data_par->inputs_count.emplace_back(cols);
    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_par.data()));
    task_data_par->outputs_count.emplace_back(out_par.size());
  }

  malyshev_lent_horizontal::TestTaskParallel taskMPI(task_data_par);
  ASSERT_EQ(taskMPI.ValidationImpl(), true);
  taskMPI.PreProcessingImpl();
  taskMPI.RunImpl();
  taskMPI.PostProcessingImpl();

  if (world.rank() == 0) {
    ASSERT_EQ(out_par, expect);
  }
}