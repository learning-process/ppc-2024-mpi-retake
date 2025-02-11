#include <gtest/gtest.h>

#include <boost/mpi/environment.hpp>
#include <random>

#include "core/util/include/util.hpp"
#include "mpi/deryabin_m_cannons_algorithm/include/ops_mpi.hpp"

TEST(deryabin_m_cannons_algorithm_mpi, test_simple_matrix) {
  boost::mpi::communicator world;
  std::vector<double> input_matrix_A{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  std::vector<double> input_matrix_B{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  std::vector<double> output_matrix_C(16, 0);
  std::vector<std::vector<double>> out_matrix_C(1, output_matrix_C);

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix_A.data()));
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix_B.data()));
    task_data_mpi->inputs_count.emplace_back(input_matrix_A.size());
    task_data_mpi->inputs_count.emplace_back(input_matrix_B.size());
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_matrix_C.data()));
    task_data_mpi->outputs_count.emplace_back(out_matrix_C.size());
  }

  deryabin_m_cannons_algorithm_mpi::CannonsAlgorithmMPITaskParallel testMpiTaskParallel(task_data_mpi);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<std::vector<double>> reference_out_matrix_C(1, output_matrix_C);

    auto task_data_seq = std::make_shared<ppc::core::TaskData>();
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix_A.data()));
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix_B.data()));
    task_data_seq->inputs_count.emplace_back(input_matrix_A.size());
    task_data_seq->inputs_count.emplace_back(input_matrix_B.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_out_matrix_C.data()));
    task_data_seq->outputs_count.emplace_back(out_matrix_C.size());

    deryabin_m_cannons_algorithm_mpi::CannonsAlgorithmMPITaskSequential testMpiTaskSequential(task_data_seq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(reference_out_matrix_C[0], out_matrix_C[0]);
  }
}

TEST(deryabin_m_cannons_algorithm_mpi, test_random_matrix) {
  boost::mpi::communicator world;
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> _distribution_(-100, 100);
  std::vector<double> input_matrix_A(16, _distribution_(gen));
  std::vector<double> input_matrix_B(16, _distribution_(gen));
  std::vector<double> output_matrix_C(16, 0);
  std::vector<std::vector<double>> out_matrix_C(1, output_matrix_C);

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix_A.data()));
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix_B.data()));
    task_data_mpi->inputs_count.emplace_back(input_matrix_A.size());
    task_data_mpi->inputs_count.emplace_back(input_matrix_B.size());
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_matrix_C.data()));
    task_data_mpi->outputs_count.emplace_back(out_matrix_C.size());
  }

  deryabin_m_cannons_algorithm_mpi::CannonsAlgorithmMPITaskParallel testMpiTaskParallel(task_data_mpi);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<std::vector<double>> reference_out_matrix_C(1, output_matrix_C);

    auto task_data_seq = std::make_shared<ppc::core::TaskData>();
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix_A.data()));
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix_B.data()));
    task_data_seq->inputs_count.emplace_back(input_matrix_A.size());
    task_data_seq->inputs_count.emplace_back(input_matrix_B.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_out_matrix_C.data()));
    task_data_seq->outputs_count.emplace_back(out_matrix_C.size());

    deryabin_m_cannons_algorithm_mpi::CannonsAlgorithmMPITaskSequential testMpiTaskSequential(task_data_seq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(reference_out_matrix_C[0], out_matrix_C[0]);
  }
}

TEST(deryabin_m_cannons_algorithm_mpi, test_gigantic_random_matrix) {
  boost::mpi::communicator world;
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> _distribution_(-1000, 1000);
  std::vector<double> input_matrix_A(1600, _distribution_(gen));
  std::vector<double> input_matrix_B(1600, _distribution_(gen));
  std::vector<double> output_matrix_C(1600, 0);
  std::vector<std::vector<double>> out_matrix_C(1, output_matrix_C);

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix_A.data()));
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix_B.data()));
    task_data_mpi->inputs_count.emplace_back(input_matrix_A.size());
    task_data_mpi->inputs_count.emplace_back(input_matrix_B.size());
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_matrix_C.data()));
    task_data_mpi->outputs_count.emplace_back(out_matrix_C.size());
  }

  deryabin_m_cannons_algorithm_mpi::CannonsAlgorithmMPITaskParallel testMpiTaskParallel(task_data_mpi);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<std::vector<double>> reference_out_matrix_C(1, output_matrix_C);

    auto task_data_seq = std::make_shared<ppc::core::TaskData>();
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix_A.data()));
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix_B.data()));
    task_data_seq->inputs_count.emplace_back(input_matrix_A.size());
    task_data_seq->inputs_count.emplace_back(input_matrix_B.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_out_matrix_C.data()));
    task_data_seq->outputs_count.emplace_back(out_matrix_C.size());

    deryabin_m_cannons_algorithm_mpi::CannonsAlgorithmMPITaskSequential testMpiTaskSequential(task_data_seq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(reference_out_matrix_C[0], out_matrix_C[0]);
  }
}

TEST(deryabin_m_cannons_algorithm_mpi, test_matrices_of_different_dimensions) {
  boost::mpi::communicator world;
  std::vector<double> input_matrix_A{1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<double> input_matrix_B{1, 2, 3, 4};
  std::vector<double> output_matrix_C(9, 0);
  std::vector<std::vector<double>> out_matrix_C(1, output_matrix_C);

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  deryabin_m_cannons_algorithm_mpi::CannonsAlgorithmMPITaskParallel testMpiTaskParallel(task_data_mpi);
  if (world.rank() == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix_A.data()));
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix_B.data()));
    task_data_mpi->inputs_count.emplace_back(input_matrix_A.size());
    task_data_mpi->inputs_count.emplace_back(input_matrix_B.size());
    ASSERT_EQ(testMpiTaskParallel.validation(), false);
  }
  if (world.rank() == 0) {
    auto task_data_seq = std::make_shared<ppc::core::TaskData>();
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix_A.data()));
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix_B.data()));
    task_data_seq->inputs_count.emplace_back(input_matrix_A.size());
    task_data_seq->inputs_count.emplace_back(input_matrix_B.size());

    deryabin_m_cannons_algorithm_mpi::CannonsAlgorithmMPITaskSequential testMpiTaskSequential(task_data_seq);
    ASSERT_EQ(testMpiTaskSequential.validation(), false);
  }
}

TEST(deryabin_m_cannons_algorithm_mpi, test_non_square_matrices) {
  boost::mpi::communicator world;
  std::vector<double> input_matrix_A{1, 2, 3, 4, 5};
  std::vector<double> input_matrix_B{1, 2, 3, 4, 5};
  std::vector<double> output_matrix_C(5, 0);
  std::vector<std::vector<double>> out_matrix_C(1, output_matrix_C);

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  deryabin_m_cannons_algorithm_mpi::CannonsAlgorithmMPITaskParallel testMpiTaskParallel(task_data_mpi);
  if (world.rank() == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix_A.data()));
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix_B.data()));
    task_data_mpi->inputs_count.emplace_back(input_matrix_A.size());
    task_data_mpi->inputs_count.emplace_back(input_matrix_B.size());
    ASSERT_EQ(testMpiTaskParallel.validation(), false);
  }
  if (world.rank() == 0) {
    auto task_data_seq = std::make_shared<ppc::core::TaskData>();
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix_A.data()));
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix_B.data()));
    task_data_seq->inputs_count.emplace_back(input_matrix_A.size());
    task_data_seq->inputs_count.emplace_back(input_matrix_B.size());

    deryabin_m_cannons_algorithm_mpi::CannonsAlgorithmMPITaskSequential testMpiTaskSequential(task_data_seq);
    ASSERT_EQ(testMpiTaskSequential.validation(), false);
  }
}
