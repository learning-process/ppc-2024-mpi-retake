#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "mpi/malyshev_v_lent_horizontal/include/ops_mpi.hpp"

namespace {
std::vector<int> GetRandomMatrix(int rows, int cols) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> matrix(rows * cols);
  for (int i = 0; i < rows * cols; i++) {
    matrix[i] = static_cast<int>(gen() % 100);
  }
  return matrix;
}

std::vector<int> GetRandomVector(int size) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vector(size);
  for (int i = 0; i < size; i++) {
    vector[i] = static_cast<int>(gen() % 100);
  }
  return vector;
}
}  // namespace

TEST(malyshev_v_lent_horizontal_mpi, test_empty_matrix) {
  boost::mpi::communicator world;

  int cols = 0;
  int rows = 0;

  std::vector<int> matrix = {};
  std::vector<int> vector = {};
  std::vector<int> out_par = {};

  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    task_data_par->inputs_count.emplace_back(matrix.size());
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t*>(vector.data()));
    task_data_par->inputs_count.emplace_back(vector.size());
    task_data_par->inputs_count.emplace_back(rows);
    task_data_par->inputs_count.emplace_back(cols);
    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_par.data()));
    task_data_par->outputs_count.emplace_back(out_par.size());
  }

  malyshev_v_lent_horizontal_mpi::MatVecMultMpi mat_vec_mult_mpi(task_data_par);
  ASSERT_TRUE(mat_vec_mult_mpi.ValidationImpl());
  mat_vec_mult_mpi.PreProcessingImpl();
  mat_vec_mult_mpi.RunImpl();
  mat_vec_mult_mpi.PostProcessingImpl();
}

TEST(malyshev_v_lent_horizontal_mpi, test_1x1_matrix) {
  boost::mpi::communicator world;

  int cols = 1;
  int rows = 1;

  std::vector<int> matrix = {2};
  std::vector<int> vector = {3};
  std::vector<int> out_par(rows, 0);

  std::vector<int> expect = {6};

  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    task_data_par->inputs_count.emplace_back(matrix.size());
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t*>(vector.data()));
    task_data_par->inputs_count.emplace_back(vector.size());
    task_data_par->inputs_count.emplace_back(rows);
    task_data_par->inputs_count.emplace_back(cols);
    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_par.data()));
    task_data_par->outputs_count.emplace_back(out_par.size());
  }

  malyshev_v_lent_horizontal_mpi::MatVecMultMpi mat_vec_mult_mpi(task_data_par);
  ASSERT_TRUE(mat_vec_mult_mpi.ValidationImpl());
  mat_vec_mult_mpi.PreProcessingImpl();
  mat_vec_mult_mpi.RunImpl();
  mat_vec_mult_mpi.PostProcessingImpl();

  if (world.rank() == 0) {
    ASSERT_EQ(out_par, expect);
  }
}

TEST(malyshev_v_lent_horizontal_mpi, test_2x2_matrix) {
  boost::mpi::communicator world;

  int cols = 2;
  int rows = 2;

  std::vector<int> matrix = {1, 2, 3, 4};
  std::vector<int> vector = {5, 6};
  std::vector<int> out_par(rows, 0);

  std::vector<int> expect = {(1 * 5) + (2 * 6), (3 * 5) + (4 * 6)};  // {17, 39}

  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    task_data_par->inputs_count.emplace_back(matrix.size());
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t*>(vector.data()));
    task_data_par->inputs_count.emplace_back(vector.size());
    task_data_par->inputs_count.emplace_back(rows);
    task_data_par->inputs_count.emplace_back(cols);
    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_par.data()));
    task_data_par->outputs_count.emplace_back(out_par.size());
  }

  malyshev_v_lent_horizontal_mpi::MatVecMultMpi mat_vec_mult_mpi(task_data_par);
  ASSERT_TRUE(mat_vec_mult_mpi.ValidationImpl());
  mat_vec_mult_mpi.PreProcessingImpl();
  mat_vec_mult_mpi.RunImpl();
  mat_vec_mult_mpi.PostProcessingImpl();

  if (world.rank() == 0) {
    ASSERT_EQ(out_par, expect);
  }
}

TEST(malyshev_v_lent_horizontal_mpi, test_3x2_matrix) {
  boost::mpi::communicator world;

  int cols = 2;
  int rows = 3;

  std::vector<int> matrix = {1, 2, 3, 4, 5, 6};
  std::vector<int> vector = {2, 3};
  std::vector<int> out_par(rows, 0);

  std::vector<int> expect = {(1 * 2) + (2 * 3), (3 * 2) + (4 * 3), (5 * 2) + (6 * 3)};  // {8, 18, 28}

  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    task_data_par->inputs_count.emplace_back(matrix.size());
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t*>(vector.data()));
    task_data_par->inputs_count.emplace_back(vector.size());
    task_data_par->inputs_count.emplace_back(rows);
    task_data_par->inputs_count.emplace_back(cols);
    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_par.data()));
    task_data_par->outputs_count.emplace_back(out_par.size());
  }

  malyshev_v_lent_horizontal_mpi::MatVecMultMpi mat_vec_mult_mpi(task_data_par);
  ASSERT_TRUE(mat_vec_mult_mpi.ValidationImpl());
  mat_vec_mult_mpi.PreProcessingImpl();
  mat_vec_mult_mpi.RunImpl();
  mat_vec_mult_mpi.PostProcessingImpl();

  if (world.rank() == 0) {
    ASSERT_EQ(out_par, expect);
  }
}

TEST(malyshev_v_lent_horizontal_mpi, test_random_matrix) {
  boost::mpi::communicator world;
  int cols = 20;
  int rows = 13;

  std::vector<int> matrix = GetRandomMatrix(rows, cols);
  std::vector<int> vector = GetRandomVector(cols);
  std::vector<int> out_par(rows, 0);

  std::vector<int> expect(rows, 0);
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      expect[i] += matrix[(i * cols) + j] * vector[j];
    }
  }

  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    task_data_par->inputs_count.emplace_back(matrix.size());
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t*>(vector.data()));
    task_data_par->inputs_count.emplace_back(vector.size());
    task_data_par->inputs_count.emplace_back(rows);
    task_data_par->inputs_count.emplace_back(cols);
    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_par.data()));
    task_data_par->outputs_count.emplace_back(out_par.size());
  }

  malyshev_v_lent_horizontal_mpi::MatVecMultMpi mat_vec_mult_mpi(task_data_par);
  ASSERT_TRUE(mat_vec_mult_mpi.ValidationImpl());
  mat_vec_mult_mpi.PreProcessingImpl();
  mat_vec_mult_mpi.RunImpl();
  mat_vec_mult_mpi.PostProcessingImpl();

  if (world.rank() == 0) {
    ASSERT_EQ(out_par, expect);
  }
}