#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "mpi/shishkarev_a_gaussian_method_horizontal_strip_pattern/include/ops_mpi.hpp"

namespace shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi {

std::vector<double> GetRandomMatrix(int sz) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_real_distribution<double> dis(-1000, 1000);
  std::vector<double> mat(sz);
  for (int i = 0; i < sz; ++i) {
    mat[i] = dis(gen);
  }
  return mat;
}

bool IsSingular(const std::vector<double>& matrix, int rows, int cols) {
  Matrix mat;
  mat.rows = rows;
  mat.cols = cols
  return Determinant(mat, matrix) == 0; 
}

}  // namespace shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi

TEST(shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi, test_empty_matrix) {
  boost::mpi::communicator world;

  const int cols = 0;
  const int rows = 0;

  std::vector<double> global_matrix;
  std::vector<double> global_res;
  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    task_data_par->inputs_count.emplace_back(global_matrix.size());
    task_data_par->inputs_count.emplace_back(cols);
    task_data_par->inputs_count.emplace_back(rows);
    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
    task_data_par->outputs_count.emplace_back(global_res.size());
    shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi::MPIGaussHorizontalParallel mpi_gauss_horizontal_parallel(
        task_data_par);
    ASSERT_FALSE(mpi_gauss_horizontal_parallel.ValidationImpl());
  }
}

TEST(shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi, test_matrix_with_one_element) {
  boost::mpi::communicator world;

  const int cols = 1;
  const int rows = 1;

  std::vector<double> global_matrix;
  std::vector<double> global_res;
  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_matrix = {1};
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    task_data_par->inputs_count.emplace_back(global_matrix.size());
    task_data_par->inputs_count.emplace_back(cols);
    task_data_par->inputs_count.emplace_back(rows);
    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
    task_data_par->outputs_count.emplace_back(global_res.size());
    shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi::MPIGaussHorizontalParallel mpi_gauss_horizontal_parallel(
        task_data_par);
    ASSERT_FALSE(mpi_gauss_horizontal_parallel.ValidationImpl());
  }
}

TEST(shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi, test_not_square_matrix) {
  boost::mpi::communicator world;

  const int cols = 5;
  const int rows = 2;

  std::vector<double> global_matrix;
  std::vector<double> global_res(cols - 1, 0);
  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_matrix = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    task_data_par->inputs_count.emplace_back(global_matrix.size());
    task_data_par->inputs_count.emplace_back(cols);
    task_data_par->inputs_count.emplace_back(rows);
    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
    task_data_par->outputs_count.emplace_back(global_res.size());
    shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi::MPIGaussHorizontalParallel mpi_gauss_horizontal_parallel(
        task_data_par);
    ASSERT_FALSE(mpi_gauss_horizontal_parallel.ValidationImpl());
  }
}

TEST(shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi, test_zero_determinant) {
  boost::mpi::communicator world;

  const int cols = 4;
  const int rows = 3;

  std::vector<double> global_matrix;
  std::vector<double> global_res(cols - 1, 0);
  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_matrix = {6, -1, 12, 3, -3, -5, -6, 9, 1, 4, 2, -1};
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    task_data_par->inputs_count.emplace_back(global_matrix.size());
    task_data_par->inputs_count.emplace_back(cols);
    task_data_par->inputs_count.emplace_back(rows);
    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
    task_data_par->outputs_count.emplace_back(global_res.size());
    shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi::MPIGaussHorizontalParallel mpi_gauss_horizontal_parallel(
        task_data_par);
    ASSERT_FALSE(mpi_gauss_horizontal_parallel.ValidationImpl());
  }
}

TEST(shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi, test_101x100) {
  boost::mpi::communicator world;

  const int cols = 101;
  const int rows = 100;

  std::vector<double> global_matrix(cols * rows);
  std::vector<double> global_res(cols - 1, 0);
  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_matrix = shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi::GetRandomMatrix(cols * rows);

    while (IsSingular(global_matrix, rows, cols)) {
      global_matrix = shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi::GetRandomMatrix(cols * rows);
    }

    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    task_data_par->inputs_count.emplace_back(global_matrix.size());
    task_data_par->inputs_count.emplace_back(cols);
    task_data_par->inputs_count.emplace_back(rows);
    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
    task_data_par->outputs_count.emplace_back(global_res.size());
  }

  shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi::MPIGaussHorizontalParallel mpi_gauss_horizontal_parallel(
      task_data_par);
  ASSERT_EQ(mpi_gauss_horizontal_parallel.ValidationImpl(), true);
  mpi_gauss_horizontal_parallel.PreProcessingImpl();
  mpi_gauss_horizontal_parallel.RunImpl();
  mpi_gauss_horizontal_parallel.PostProcessingImpl();

  if (world.rank() == 0) {
    std::vector<double> reference_res(cols - 1, 0);

    std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    task_data_seq->inputs_count.emplace_back(global_matrix.size());
    task_data_seq->inputs_count.emplace_back(cols);
    task_data_seq->inputs_count.emplace_back(rows);
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_res.data()));
    task_data_seq->outputs_count.emplace_back(reference_res.size());

    shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi::MPIGaussHorizontalSequential
        mpi_gauss_horizontal_sequential(task_data_seq);
    ASSERT_EQ(mpi_gauss_horizontal_sequential.ValidationImpl(), true);
    mpi_gauss_horizontal_sequential.PreProcessingImpl();
    mpi_gauss_horizontal_sequential.RunImpl();
    mpi_gauss_horizontal_sequential.PostProcessingImpl();

    for (int i = 0; i < cols - 1; ++i) {
      ASSERT_NEAR(global_res[i], reference_res[i], 1e-6);
    }
  }
}

TEST(shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi, test_201x200) {
  boost::mpi::communicator world;

  const int cols = 201;
  const int rows = 200;

  std::vector<double> global_matrix(cols * rows);
  std::vector<double> global_res(cols - 1, 0);
  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_matrix = shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi::GetRandomMatrix(cols * rows);

    while (IsSingular(global_matrix, rows, cols)) {
      global_matrix = shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi::GetRandomMatrix(cols * rows);
    }

    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    task_data_par->inputs_count.emplace_back(global_matrix.size());
    task_data_par->inputs_count.emplace_back(cols);
    task_data_par->inputs_count.emplace_back(rows);
    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
    task_data_par->outputs_count.emplace_back(global_res.size());
  }

  shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi::MPIGaussHorizontalParallel mpi_gauss_horizontal_parallel(
      task_data_par);
  ASSERT_EQ(mpi_gauss_horizontal_parallel.ValidationImpl(), true);
  mpi_gauss_horizontal_parallel.PreProcessingImpl();
  mpi_gauss_horizontal_parallel.RunImpl();
  mpi_gauss_horizontal_parallel.PostProcessingImpl();

  if (world.rank() == 0) {
    std::vector<double> reference_res(cols - 1, 0);

    std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    task_data_seq->inputs_count.emplace_back(global_matrix.size());
    task_data_seq->inputs_count.emplace_back(cols);
    task_data_seq->inputs_count.emplace_back(rows);
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_res.data()));
    task_data_seq->outputs_count.emplace_back(reference_res.size());

    shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi::MPIGaussHorizontalSequential
        mpi_gauss_horizontal_sequential(task_data_seq);
    ASSERT_EQ(mpi_gauss_horizontal_sequential.ValidationImpl(), true);
    mpi_gauss_horizontal_sequential.PreProcessingImpl();
    mpi_gauss_horizontal_sequential.RunImpl();
    mpi_gauss_horizontal_sequential.PostProcessingImpl();

    for (int i = 0; i < cols - 1; ++i) {
      ASSERT_NEAR(global_res[i], reference_res[i], 1e-6);
    }
  }
}
