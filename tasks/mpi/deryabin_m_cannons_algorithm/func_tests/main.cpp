#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "mpi/deryabin_m_cannons_algorithm/include/ops_mpi.hpp"

TEST(deryabin_m_cannons_algorithm_mpi, test_simple_matrix) {
  boost::mpi::communicator world;
  std::vector<double> input_matrix_a{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  std::vector<double> input_matrix_b{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  std::vector<double> output_matrix_c(16, 0);
  std::vector<std::vector<double>> out_matrix_c(1, output_matrix_c);
  std::vector<double> true_solution{90, 100, 110, 120, 202, 228, 254, 280, 314, 356, 398, 440, 426, 484, 542, 600};

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix_a.data()));
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix_b.data()));
    task_data_mpi->inputs_count.emplace_back(input_matrix_a.size());
    task_data_mpi->inputs_count.emplace_back(input_matrix_b.size());
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_matrix_c.data()));
    task_data_mpi->outputs_count.emplace_back(out_matrix_c.size());
  }

  deryabin_m_cannons_algorithm_mpi::CannonsAlgorithmMPITaskParallel test_mpi_task_parallel(task_data_mpi);
  ASSERT_EQ(test_mpi_task_parallel.Validation(), true);
  test_mpi_task_parallel.PreProcessing();
  test_mpi_task_parallel.Run();
  test_mpi_task_parallel.PostProcessing();

  if (world.rank() == 0) {
    ASSERT_EQ(true_solution, out_matrix_c[0]);
  }
}
