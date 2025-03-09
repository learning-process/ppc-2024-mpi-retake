#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <cmath>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "mpi/shkurinskaya_e_fox_matrix_mult/include/ops_sec.hpp"

namespace shkurinskaya_e_fox_mat_mul_mpi {
std::vector<double> GetRandomMatrix(int rows, int cols) {
  std::vector<double> result(rows * cols);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(-50.0, 50.0);

  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      result[(i * cols) + j] = dis(gen);
    }
  }

  return result;
}

void SimpleMult(std::vector<double> &in1, std::vector<double> &in2, std::vector<double> &ans, int matrix_size) {
  for (int i = 0; i < matrix_size; ++i) {
    for (int j = 0; j < matrix_size; ++j) {
      for (int k = 0; k < matrix_size; ++k) {
        ans[(i * matrix_size) + j] += in1[(i * matrix_size) + k] * in2[(k * matrix_size) + j];
      }
    }
  }
}

}  // namespace shkurinskaya_e_fox_mat_mul_mpi

TEST(shkurinskaya_e_fox_mat_mul_mpi, small_matrix) {
  boost::mpi::communicator world;

  int matrix_size = 4;
  std::vector<double> in1;
  std::vector<double> in2;
  std::vector<double> out;
  std::vector<double> ans;
  auto test_info = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    in1 = shkurinskaya_e_fox_mat_mul_mpi::GetRandomMatrix(matrix_size, matrix_size);
    in2 = shkurinskaya_e_fox_mat_mul_mpi::GetRandomMatrix(matrix_size, matrix_size);
    out.resize(matrix_size * matrix_size);
    ans.resize(matrix_size * matrix_size);

    shkurinskaya_e_fox_mat_mul_mpi::SimpleMult(in1, in2, ans, matrix_size);
    // create task data
    test_info->inputs.emplace_back(reinterpret_cast<uint8_t *>(in1.data()));
    test_info->inputs.emplace_back(reinterpret_cast<uint8_t *>(in2.data()));
    test_info->inputs_count.emplace_back(matrix_size);
    test_info->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    test_info->outputs_count.emplace_back(matrix_size);
  }
  // Create Task
  shkurinskaya_e_fox_mat_mul_mpi::FoxMatMulMPI job(test_info);
  ASSERT_EQ(job.Validation(), true);
  job.PreProcessing();
  job.Run();
  job.PostProcessing();
  if (world.rank() == 0) {
    for (int i = 0; i < (int)ans.size(); ++i) {
      ASSERT_NEAR(ans[i], out[i], 1);
    }
  }
}

TEST(shkurinskaya_e_fox_mat_mul_mpi, big_matrix) {
  boost::mpi::communicator world;

  int matrix_size = 72;
  std::vector<double> in1;
  std::vector<double> in2;
  std::vector<double> out;
  std::vector<double> ans;
  auto test_info = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    in1 = shkurinskaya_e_fox_mat_mul_mpi::GetRandomMatrix(matrix_size, matrix_size);
    in2 = shkurinskaya_e_fox_mat_mul_mpi::GetRandomMatrix(matrix_size, matrix_size);
    out.resize(matrix_size * matrix_size);
    ans.resize(matrix_size * matrix_size);

    shkurinskaya_e_fox_mat_mul_mpi::SimpleMult(in1, in2, ans, matrix_size);

    // create task data
    test_info->inputs.emplace_back(reinterpret_cast<uint8_t *>(in1.data()));
    test_info->inputs.emplace_back(reinterpret_cast<uint8_t *>(in2.data()));
    test_info->inputs_count.emplace_back(matrix_size);
    test_info->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    test_info->outputs_count.emplace_back(matrix_size);
  }
  // Create Task
  shkurinskaya_e_fox_mat_mul_mpi::FoxMatMulMPI job(test_info);
  ASSERT_EQ(job.Validation(), true);
  job.PreProcessing();
  job.Run();
  job.PostProcessing();
  if (world.rank() == 0) {
    for (int i = 0; i < (int)ans.size(); ++i) {
      ASSERT_NEAR(ans[i], out[i], 1);
    }
  }
}

TEST(shkurinskaya_e_fox_mat_mul_mpi, validation_false_1) {
  boost::mpi::communicator world;

  int matrix_size = 0;
  std::vector<double> in1;
  std::vector<double> in2;
  std::vector<double> out;
  std::vector<double> ans;
  auto test_info = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    // create task data
    test_info->inputs.emplace_back(reinterpret_cast<uint8_t *>(in1.data()));
    test_info->inputs.emplace_back(reinterpret_cast<uint8_t *>(in2.data()));
    test_info->inputs_count.emplace_back(matrix_size);
    test_info->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    test_info->outputs_count.emplace_back(matrix_size);
  }
  // Create Task
  shkurinskaya_e_fox_mat_mul_mpi::FoxMatMulMPI job(test_info);
  if (world.rank() == 0) {
    ASSERT_EQ(job.Validation(), false);
  }
}
TEST(shkurinskaya_e_fox_mat_mul_mpi, validation_false_2) {
  boost::mpi::communicator world;

  int matrix_size = 1;
  std::vector<double> in1;
  std::vector<double> in2;
  std::vector<double> out;
  std::vector<double> ans;
  auto test_info = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    // create task data
    test_info->inputs.emplace_back(reinterpret_cast<uint8_t *>(in1.data()));
    test_info->inputs.emplace_back(reinterpret_cast<uint8_t *>(in2.data()));
    test_info->inputs_count.emplace_back(matrix_size);
    test_info->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    test_info->outputs_count.emplace_back(2);
  }
  // Create Task
  shkurinskaya_e_fox_mat_mul_mpi::FoxMatMulMPI job(test_info);
  if (world.rank() == 0) {
    ASSERT_EQ(job.Validation(), false);
  }
}
