#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "mpi/Konstantinov_I_Gauss_Jordan_method/include/ops_mpi.hpp"

namespace konstantinov_i_gauss_jordan_method_mpi {
namespace {
std::vector<double> GenerateInvertibleMatrix(int size) {
  std::vector<double> matrix(size * (size + 1));
  std::random_device rd;
  std::mt19937 gen(rd());
  double lower_limit = -100.0;
  double upper_limit = 100.0;
  std::uniform_real_distribution<> dist(lower_limit, upper_limit);

  for (int i = 0; i < size; ++i) {
    double row_sum = 0.0;
    double diag = (i * (size + 1) + i);
    for (int j = 0; j < size + 1; ++j) {
      if (i != j) {
        matrix[(i * (size + 1)) + j] = dist(gen);
        row_sum += std::abs(matrix[(i * (size + 1)) + j]);
      }
    }
    matrix[diag] = static_cast<size_t>(row_sum + 1);
  }

  return matrix;
}
}  // namespace
}  // namespace konstantinov_i_gauss_jordan_method_mpi

TEST(Konstantinov_i_gauss_jordan_method_mpi, Test_2x2) {
  boost::mpi::communicator world;
  int size = 2;
  std::vector<double> matrix = {2, 3, 5, 4, 1, 6};
  std::vector<double> output_data(size, 0.0);
  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(&size));
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    task_data_par->inputs_count.emplace_back(matrix.size() / (size + 1));
    task_data_par->inputs_count.emplace_back(matrix.size());
    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_data.data()));
    task_data_par->outputs_count.emplace_back(output_data.size());
  }
  konstantinov_i_gauss_jordan_method_mpi::GaussJordanMethodMpi test_mpi(task_data_par);
  ASSERT_EQ(test_mpi.ValidationImpl(), true);
  test_mpi.PreProcessingImpl();
  test_mpi.RunImpl();
  test_mpi.PostProcessingImpl();

  if (world.rank() == 0) {
    std::vector<double> reference_data(size, 0.0);
    std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&size));
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    task_data_seq->inputs_count.emplace_back(matrix.size() / (size + 1));
    task_data_seq->inputs_count.emplace_back(matrix.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_data.data()));
    task_data_seq->outputs_count.emplace_back(reference_data.size());

    konstantinov_i_gauss_jordan_method_mpi::GaussJordanMethodSeq test_seq_mpi(task_data_seq);
    ASSERT_EQ(test_seq_mpi.ValidationImpl(), true);
    test_seq_mpi.PreProcessingImpl();
    test_seq_mpi.RunImpl();
    test_seq_mpi.PostProcessingImpl();

    for (int i = 0; i < size; ++i) {
      ASSERT_EQ(reference_data[i], output_data[i]);
    }
  }
}

TEST(Konstantinov_i_gauss_jordan_method_mpi, Test_5x5) {
  boost::mpi::communicator world;
  int size = 2;
  std::vector<double> matrix = konstantinov_i_gauss_jordan_method_mpi::GenerateInvertibleMatrix(size);
  std::vector<double> output_data(size, 0.0);
  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(&size));
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    task_data_par->inputs_count.emplace_back(matrix.size() / (size + 1));
    task_data_par->inputs_count.emplace_back(matrix.size());
    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_data.data()));
    task_data_par->outputs_count.emplace_back(output_data.size());
  }

  konstantinov_i_gauss_jordan_method_mpi::GaussJordanMethodMpi test_mpi(task_data_par);
  ASSERT_EQ(test_mpi.ValidationImpl(), true);
  test_mpi.PreProcessingImpl();
  test_mpi.RunImpl();
  test_mpi.PostProcessingImpl();

  if (world.rank() == 0) {
    std::vector<double> reference_data(size, 0.0);
    std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&size));
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    task_data_seq->inputs_count.emplace_back(matrix.size() / (size + 1));
    task_data_seq->inputs_count.emplace_back(matrix.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_data.data()));
    task_data_seq->outputs_count.emplace_back(reference_data.size());

    konstantinov_i_gauss_jordan_method_mpi::GaussJordanMethodSeq test_seq_mpi(task_data_seq);
    ASSERT_EQ(test_seq_mpi.ValidationImpl(), true);
    test_seq_mpi.PreProcessingImpl();
    test_seq_mpi.RunImpl();
    test_seq_mpi.PostProcessingImpl();

    for (int i = 0; i < size; ++i) {
      ASSERT_EQ(reference_data[i], output_data[i]);
    }
  }
}

TEST(Konstantinov_i_gauss_jordan_method_mpi, Test_50x50) {
  boost::mpi::communicator world;
  int size = 50;
  std::vector<double> matrix = konstantinov_i_gauss_jordan_method_mpi::GenerateInvertibleMatrix(size);
  std::vector<double> output_data(size, 0.0);
  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(&size));
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    task_data_par->inputs_count.emplace_back(matrix.size() / (size + 1));
    task_data_par->inputs_count.emplace_back(matrix.size());
    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_data.data()));
    task_data_par->outputs_count.emplace_back(output_data.size());
  }

  konstantinov_i_gauss_jordan_method_mpi::GaussJordanMethodMpi test_mpi(task_data_par);
  ASSERT_EQ(test_mpi.ValidationImpl(), true);
  test_mpi.PreProcessingImpl();
  test_mpi.RunImpl();
  test_mpi.PostProcessingImpl();

  if (world.rank() == 0) {
    std::vector<double> reference_data(size, 0.0);
    std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&size));
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    task_data_seq->inputs_count.emplace_back(matrix.size() / (size + 1));
    task_data_seq->inputs_count.emplace_back(matrix.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_data.data()));
    task_data_seq->outputs_count.emplace_back(reference_data.size());

    konstantinov_i_gauss_jordan_method_mpi::GaussJordanMethodSeq test_seq_mpi(task_data_seq);
    ASSERT_EQ(test_seq_mpi.ValidationImpl(), true);
    test_seq_mpi.PreProcessingImpl();
    test_seq_mpi.RunImpl();
    test_seq_mpi.PostProcessingImpl();

    for (int i = 0; i < size; ++i) {
      ASSERT_EQ(reference_data[i], output_data[i]);
    }
  }
}

TEST(Konstantinov_i_gauss_jordan_method_mpi, Test_invalid_data) {
  boost::mpi::communicator world;
  int size = 2;
  std::vector<double> matrix = {2, 3, 5, 4, 1, 6, 8};
  std::vector<double> output_data(size, 0.0);
  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(&size));
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    task_data_par->inputs_count.emplace_back(matrix.size() / (size + 1));
    task_data_par->inputs_count.emplace_back(matrix.size());
    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_data.data()));
    task_data_par->outputs_count.emplace_back(output_data.size());
  }

  if (world.rank() == 0) {
    konstantinov_i_gauss_jordan_method_mpi::GaussJordanMethodMpi test_mpi(task_data_par);
    ASSERT_FALSE(test_mpi.ValidationImpl());
  }

  if (world.rank() == 0) {
    std::vector<double> reference_data(size, 0.0);
    std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&size));
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    task_data_seq->inputs_count.emplace_back(matrix.size() / (size + 1));
    task_data_seq->inputs_count.emplace_back(matrix.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_data.data()));
    task_data_seq->outputs_count.emplace_back(reference_data.size());

    konstantinov_i_gauss_jordan_method_mpi::GaussJordanMethodSeq test_seq_mpi(task_data_seq);
    ASSERT_FALSE(test_seq_mpi.ValidationImpl());
  }
}

TEST(Konstantinov_i_gauss_jordan_method_mpi, Test_not_enough_data) {
  boost::mpi::communicator world;
  int size = 2;
  std::vector<double> matrix = {2, 3, 5};
  std::vector<double> output_data(size, 0.0);
  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(&size));
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    task_data_par->inputs_count.emplace_back(matrix.size() / (size + 1));
    task_data_par->inputs_count.emplace_back(matrix.size());
    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_data.data()));
    task_data_par->outputs_count.emplace_back(output_data.size());
  }

  if (world.rank() == 0) {
    konstantinov_i_gauss_jordan_method_mpi::GaussJordanMethodMpi test_mpi(task_data_par);
    ASSERT_FALSE(test_mpi.ValidationImpl());
  } else {
    ASSERT_TRUE(true) << "Process " << world.rank() << " completed successfully.";
  }

  if (world.rank() == 0) {
    std::vector<double> reference_data(size, 0.0);
    std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&size));
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    task_data_seq->inputs_count.emplace_back(matrix.size() / (size + 1));
    task_data_seq->inputs_count.emplace_back(matrix.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_data.data()));
    task_data_seq->outputs_count.emplace_back(reference_data.size());

    konstantinov_i_gauss_jordan_method_mpi::GaussJordanMethodSeq test_seq_mpi(task_data_seq);
    ASSERT_FALSE(test_seq_mpi.ValidationImpl());
  }
}

TEST(Konstantinov_i_gauss_jordan_method_mpi, Test_zero_diag) {
  boost::mpi::communicator world;
  int size = 3;
  std::vector<double> matrix = {0, 1, 1, 1, 2, 1, 2, 2, 2, 2, 4, 3};
  std::vector<double> output_data(size, 0.0);
  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(&size));
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    task_data_par->inputs_count.emplace_back(matrix.size() / (size + 1));
    task_data_par->inputs_count.emplace_back(matrix.size());
    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_data.data()));
    task_data_par->outputs_count.emplace_back(output_data.size());
  }

  if (world.rank() == 0) {
    konstantinov_i_gauss_jordan_method_mpi::GaussJordanMethodMpi test_mpi(task_data_par);
    ASSERT_FALSE(test_mpi.ValidationImpl());
  } else {
    ASSERT_TRUE(true) << "Process " << world.rank() << " completed successfully.";
  }
  if (world.rank() == 0) {
    std::vector<double> reference_data(size, 0.0);
    std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&size));
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    task_data_seq->inputs_count.emplace_back(matrix.size() / (size + 1));
    task_data_seq->inputs_count.emplace_back(matrix.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_data.data()));
    task_data_seq->outputs_count.emplace_back(reference_data.size());

    konstantinov_i_gauss_jordan_method_mpi::GaussJordanMethodSeq test_seq_mpi(task_data_seq);
    ASSERT_FALSE(test_seq_mpi.ValidationImpl());
  }
}