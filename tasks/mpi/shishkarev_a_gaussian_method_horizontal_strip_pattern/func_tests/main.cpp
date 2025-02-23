#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <random>
#include <vector>

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

}  // namespace shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi

TEST(shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi, test_empty_matrix) {
  boost::mpi::communicator world_;

  const int cols_ = 0;
  const int rows_ = 0;

  std::vector<double> global_matrix;
  std::vector<double> global_res;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world_.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataPar->inputs_count.emplace_back(global_matrix.size());
    taskDataPar->inputs_count.emplace_back(cols_);
    taskDataPar->inputs_count.emplace_back(rows_);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
    shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi::MPIGaussHorizontalParallel MPIGaussHorizontalParallel(
        taskDataPar);
    ASSERT_FALSE(MPIGaussHorizontalParallel.ValidationImpl());
  }
}

TEST(shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi, test_matrix_with_one_element) {
  boost::mpi::communicator world_;

  const int cols_ = 1;
  const int rows_ = 1;

  std::vector<double> global_matrix;
  std::vector<double> global_res;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world_.rank() == 0) {
    global_matrix = {1};
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataPar->inputs_count.emplace_back(global_matrix.size());
    taskDataPar->inputs_count.emplace_back(cols_);
    taskDataPar->inputs_count.emplace_back(rows_);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
    shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi::MPIGaussHorizontalParallel MPIGaussHorizontalParallel(
        taskDataPar);
    ASSERT_FALSE(MPIGaussHorizontalParallel.ValidationImpl());
  }
}

TEST(shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi, test_not_square_matrix) {
  boost::mpi::communicator world_;

  const int cols_ = 5;
  const int rows_ = 2;

  std::vector<double> global_matrix;
  std::vector<double> global_res(cols_ - 1, 0);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world_.rank() == 0) {
    global_matrix = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataPar->inputs_count.emplace_back(global_matrix.size());
    taskDataPar->inputs_count.emplace_back(cols_);
    taskDataPar->inputs_count.emplace_back(rows_);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
    shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi::MPIGaussHorizontalParallel MPIGaussHorizontalParallel(
        taskDataPar);
    ASSERT_FALSE(MPIGaussHorizontalParallel.ValidationImpl());
  }
}

TEST(shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi, test_zero_determinant) {
  boost::mpi::communicator world_;

  const int cols_ = 4;
  const int rows_ = 3;

  std::vector<double> global_matrix;
  std::vector<double> global_res(cols_ - 1, 0);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world_.rank() == 0) {
    global_matrix = {6, -1, 12, 3, -3, -5, -6, 9, 1, 4, 2, -1};
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataPar->inputs_count.emplace_back(global_matrix.size());
    taskDataPar->inputs_count.emplace_back(cols_);
    taskDataPar->inputs_count.emplace_back(rows_);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
    shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi::MPIGaussHorizontalParallel MPIGaussHorizontalParallel(
        taskDataPar);
    ASSERT_FALSE(MPIGaussHorizontalParallel.ValidationImpl());
  }
}

TEST(shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi, test_101x100) {
  boost::mpi::communicator world_;

  const int cols_ = 101;
  const int rows_ = 100;

  std::vector<double> global_matrix(cols_ * rows_);
  std::vector<double> global_res(cols_ - 1, 0);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world_.rank() == 0) {
    global_matrix = shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi::GetRandomMatrix(cols_ * rows_);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataPar->inputs_count.emplace_back(global_matrix.size());
    taskDataPar->inputs_count.emplace_back(cols_);
    taskDataPar->inputs_count.emplace_back(rows_);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
  }

  shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi::MPIGaussHorizontalParallel MPIGaussHorizontalParallel(
      taskDataPar);
  ASSERT_EQ(MPIGaussHorizontalParallel.ValidationImpl(), true);
  MPIGaussHorizontalParallel.PreProcessingImpl();
  MPIGaussHorizontalParallel.RunImpl();
  MPIGaussHorizontalParallel.PostProcessingImpl();

  if (world_.rank() == 0) {
    std::vector<double> reference_res(cols_ - 1, 0);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataSeq->inputs_count.emplace_back(global_matrix.size());
    taskDataSeq->inputs_count.emplace_back(cols_);
    taskDataSeq->inputs_count.emplace_back(rows_);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_res.data()));
    taskDataSeq->outputs_count.emplace_back(reference_res.size());

    shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi::MPIGaussHorizontalSequential
        MPIGaussHorizontalSequential(taskDataSeq);
    ASSERT_EQ(MPIGaussHorizontalSequential.ValidationImpl(), true);
    MPIGaussHorizontalSequential.PreProcessingImpl();
    MPIGaussHorizontalSequential.RunImpl();
    MPIGaussHorizontalSequential.PostProcessingImpl();

    for (int i = 0; i < cols_ - 1; ++i) {
      ASSERT_NEAR(global_res[i], reference_res[i], 1e-6);
    }
  }
}

TEST(shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi, test_201x200) {
  boost::mpi::communicator world_;

  const int cols_ = 201;
  const int rows_ = 200;

  std::vector<double> global_matrix(cols_ * rows_);
  std::vector<double> global_res(cols_ - 1, 0);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world_.rank() == 0) {
    global_matrix = shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi::GetRandomMatrix(cols_ * rows_);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataPar->inputs_count.emplace_back(global_matrix.size());
    taskDataPar->inputs_count.emplace_back(cols_);
    taskDataPar->inputs_count.emplace_back(rows_);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
  }

  shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi::MPIGaussHorizontalParallel MPIGaussHorizontalParallel(
      taskDataPar);
  ASSERT_EQ(MPIGaussHorizontalParallel.ValidationImpl(), true);
  MPIGaussHorizontalParallel.PreProcessingImpl();
  MPIGaussHorizontalParallel.RunImpl();
  MPIGaussHorizontalParallel.PostProcessingImpl();

  if (world_.rank() == 0) {
    std::vector<double> reference_res(cols_ - 1, 0);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataSeq->inputs_count.emplace_back(global_matrix.size());
    taskDataSeq->inputs_count.emplace_back(cols_);
    taskDataSeq->inputs_count.emplace_back(rows_);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_res.data()));
    taskDataSeq->outputs_count.emplace_back(reference_res.size());

    shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi::MPIGaussHorizontalSequential
        MPIGaussHorizontalSequential(taskDataSeq);
    ASSERT_EQ(MPIGaussHorizontalSequential.ValidationImpl(), true);
    MPIGaussHorizontalSequential.PreProcessingImpl();
    MPIGaussHorizontalSequential.RunImpl();
    MPIGaussHorizontalSequential.PostProcessingImpl();

    for (int i = 0; i < cols_ - 1; ++i) {
      ASSERT_NEAR(global_res[i], reference_res[i], 1e-6);
    }
  }
}
