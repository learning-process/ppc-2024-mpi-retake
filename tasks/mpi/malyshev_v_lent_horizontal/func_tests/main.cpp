#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "mpi/malyshev_v_lent_horizontal/include/ops_mpi.hpp"

TEST(malyshev_v_lent_horizontal, test_empty_matrix) {
  boost::mpi::communicator world;

  int cols = 0;
  int rows = 0;

  std::vector<int> matrix = {};
  std::vector<int> vector = {};
  std::vector<int> out_par = {};

  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    task_data_par->inputs_count.emplace_back(matrix.size());
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(vector.data()));
    task_data_par->inputs_count.emplace_back(vector.size());
    task_data_par->inputs_count.emplace_back(rows);
    task_data_par->inputs_count.emplace_back(cols);
    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_par.data()));
    task_data_par->outputs_count.emplace_back(out_par.size());
  }

  malyshev_v_lent_horizontal::MatVecMultMpi mat_vec_mult_mpi(task_data_par);
  ASSERT_EQ(mat_vec_mult_mpi.ValidationImpl(), true);
  mat_vec_mult_mpi.PreProcessingImpl();
  mat_vec_mult_mpi.RunImpl();
  mat_vec_mult_mpi.PostProcessingImpl();
}

TEST(malyshev_v_lent_horizontal, test_1x1_matrix) {
  boost::mpi::communicator world;

  int cols = 1;
  int rows = 1;

  std::vector<int> matrix = {2};
  std::vector<int> vector = {3};
  std::vector<int> out_par(rows, 0);

  std::vector<int> expect = {6};

  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    task_data_par->inputs_count.emplace_back(matrix.size());
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(vector.data()));
    task_data_par->inputs_count.emplace_back(vector.size());
    task_data_par->inputs_count.emplace_back(rows);
    task_data_par->inputs_count.emplace_back(cols);
    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_par.data()));
    task_data_par->outputs_count.emplace_back(out_par.size());
  }

  malyshev_v_lent_horizontal::MatVecMultMpi mat_vec_mult_mpi(task_data_par);
  ASSERT_EQ(mat_vec_mult_mpi.ValidationImpl(), true);
  mat_vec_mult_mpi.PreProcessingImpl();
  mat_vec_mult_mpi.RunImpl();
  mat_vec_mult_mpi.PostProcessingImpl();

  if (world.rank() == 0) {
    ASSERT_EQ(out_par, expect);
  }
}

TEST(malyshev_v_lent_horizontal, test_random_matrix) {
  boost::mpi::communicator world;
  int cols = 20;
  int rows = 13;

  std::vector<int> matrix = malyshev_v_lent_horizontal::GetRandomMatrix(rows, cols);
  std::vector<int> vector = malyshev_v_lent_horizontal::GetRandomVector(cols);
  std::vector<int> out_par(rows, 0);

  std::vector<int> expect(rows, 0);
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      expect[i] += matrix[(i * cols) + j] * vector[j];
    }
  }

  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    task_data_par->inputs_count.emplace_back(matrix.size());
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(vector.data()));
    task_data_par->inputs_count.emplace_back(vector.size());
    task_data_par->inputs_count.emplace_back(rows);
    task_data_par->inputs_count.emplace_back(cols);
    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_par.data()));
    task_data_par->outputs_count.emplace_back(out_par.size());
  }

  malyshev_v_lent_horizontal::MatVecMultMpi mat_vec_mult_mpi(task_data_par);
  ASSERT_EQ(mat_vec_mult_mpi.ValidationImpl(), true);
  mat_vec_mult_mpi.PreProcessingImpl();
  mat_vec_mult_mpi.RunImpl();
  mat_vec_mult_mpi.PostProcessingImpl();

  if (world.rank() == 0) {
    ASSERT_EQ(out_par, expect);
  }
}