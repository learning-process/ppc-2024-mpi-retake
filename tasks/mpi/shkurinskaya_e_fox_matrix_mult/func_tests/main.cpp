#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"
#include "core/util/include/util.hpp"
#include "mpi/shkurinskaya_e_fox_matrix_mult/include/ops_sec.hpp"

TEST(shkurinskaya_e_fox_mat_mul_mpi, small_matrix) {
  boost::mpi::communicator world;
  int root = sqrt(world.size());
  if (root * root != world.size()) {
    GTEST_SKIP();
  }

  int matrix_size = 4;
  std::vector<double> in1, in2, out, ans;
  auto test_info = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    in1 = shkurinskaya_e_fox_mat_mul_mpi::getRandomMatrix(matrix_size, matrix_size);
    in2 = shkurinskaya_e_fox_mat_mul_mpi::getRandomMatrix(matrix_size, matrix_size);
    out.resize(matrix_size * matrix_size);
    ans.resize(matrix_size * matrix_size);

    for (int i = 0; i < matrix_size; ++i) {
      for (int j = 0; j < matrix_size; ++j) {
        for (int k = 0; k < matrix_size; ++k) {
          ans[i * matrix_size + j] += in1[i * matrix_size + k] * in2[k * matrix_size + j];
        }
      }
    }

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
    for (size_t i = 0; i < ans.size(); ++i) {
      ASSERT_NEAR(ans[i], out[i], 1);
    }
  }
}

TEST(shkurinskaya_e_fox_mat_mul_mpi, big_matrix) {
  boost::mpi::communicator world;
  int root = sqrt(world.size());
  if (root * root != world.size()) {
    GTEST_SKIP();
  }

  int matrix_size = 72;
  std::vector<double> in1, in2, out, ans;
  auto test_info = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    in1 = shkurinskaya_e_fox_mat_mul_mpi::getRandomMatrix(matrix_size, matrix_size);
    in2 = shkurinskaya_e_fox_mat_mul_mpi::getRandomMatrix(matrix_size, matrix_size);
    out.resize(matrix_size * matrix_size);
    ans.resize(matrix_size * matrix_size);

    for (int i = 0; i < matrix_size; ++i) {
      for (int j = 0; j < matrix_size; ++j) {
        for (int k = 0; k < matrix_size; ++k) {
          ans[i * matrix_size + j] += in1[i * matrix_size + k] * in2[k * matrix_size + j];
        }
      }
    }

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
    for (size_t i = 0; i < ans.size(); ++i) {
      ASSERT_NEAR(ans[i], out[i], 1);
    }
  }
}

TEST(shkurinskaya_e_fox_mat_mul_mpi, validation_false_1) {
  boost::mpi::communicator world;

  int matrix_size = 0;
  std::vector<double> in1, in2, out, ans;
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
  std::vector<double> in1, in2, out, ans;
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
