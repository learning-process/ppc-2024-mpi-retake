#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "mpi/deryabin_m_cannons_algorithm/include/ops_mpi.hpp"

TEST(deryabin_m_cannons_algorithm_mpi, test_matrices_of_different_dimensions) {
  boost::mpi::communicator world;
  std::vector<double> input_matrix_a{1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<double> input_matrix_b{1, 2, 3, 4};
  std::vector<double> output_matrix_c(9, 0);
  std::vector<std::vector<double>> out_matrix_c(1, output_matrix_c);

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  deryabin_m_cannons_algorithm_mpi::CannonsAlgorithmMPITaskParallel test_mpi_task_parallel(task_data_mpi);
  if (world.rank() == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix_a.data()));
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix_b.data()));
    task_data_mpi->inputs_count.emplace_back(input_matrix_a.size());
    task_data_mpi->inputs_count.emplace_back(input_matrix_b.size());
    ASSERT_EQ(test_mpi_task_parallel.Validation(), false);
  }
  if (world.rank() == 0) {
    auto task_data_seq = std::make_shared<ppc::core::TaskData>();
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix_a.data()));
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix_b.data()));
    task_data_seq->inputs_count.emplace_back(input_matrix_a.size());
    task_data_seq->inputs_count.emplace_back(input_matrix_b.size());

    deryabin_m_cannons_algorithm_mpi::CannonsAlgorithmMPITaskSequential test_mpi_task_sequential(task_data_seq);
    ASSERT_EQ(test_mpi_task_sequential.Validation(), false);
  }
}

TEST(deryabin_m_cannons_algorithm_mpi, test_non_square_matrices) {
  boost::mpi::communicator world;
  std::vector<double> input_matrix_a{1, 2, 3, 4, 5};
  std::vector<double> input_matrix_b{1, 2, 3, 4, 5};
  std::vector<double> output_matrix_c(5, 0);
  std::vector<std::vector<double>> out_matrix_c(1, output_matrix_c);

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  deryabin_m_cannons_algorithm_mpi::CannonsAlgorithmMPITaskParallel test_mpi_task_parallel(task_data_mpi);
  if (world.rank() == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix_a.data()));
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix_b.data()));
    task_data_mpi->inputs_count.emplace_back(input_matrix_a.size());
    task_data_mpi->inputs_count.emplace_back(input_matrix_b.size());
    ASSERT_EQ(test_mpi_task_parallel.Validation(), false);
  }
  if (world.rank() == 0) {
    auto task_data_seq = std::make_shared<ppc::core::TaskData>();
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix_a.data()));
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix_b.data()));
    task_data_seq->inputs_count.emplace_back(input_matrix_a.size());
    task_data_seq->inputs_count.emplace_back(input_matrix_b.size());

    deryabin_m_cannons_algorithm_mpi::CannonsAlgorithmMPITaskSequential test_mpi_task_sequential(task_data_seq);
    ASSERT_EQ(test_mpi_task_sequential.Validation(), false);
  }
}
