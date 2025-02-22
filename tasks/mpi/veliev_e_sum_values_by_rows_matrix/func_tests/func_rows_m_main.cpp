#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "mpi/veliev_e_sum_values_by_rows_matrix/include/rows_m_header.hpp"

TEST(veliev_e_sum_values_by_rows_matrix_mpi, Test_matr_0x0) {
  std::vector base_input = {0, 0, 0};

  // Create data
  boost::mpi::communicator world;
  std::vector<int> arr(base_input[0]);

  std::vector<int> out(base_input[1]);
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    veliev_e_sum_values_by_rows_matrix_mpi::GetRndMatrix(arr);
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(base_input.data()));  // num_elem + rows + cols
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(arr.data()));
    task_data->inputs_count.emplace_back(3);
    task_data->inputs_count.emplace_back(arr.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data->outputs_count.emplace_back(base_input[1]);
  }

  veliev_e_sum_values_by_rows_matrix_mpi::SumValuesByRowsMatrixMpi test1(task_data);

  ASSERT_EQ(test1.Validation(), true);
  test1.PreProcessing();
  test1.Run();
  test1.PostProcessing();
  if (world.rank() == 0) {
    std::vector<int> ref_for_check;
    veliev_e_sum_values_by_rows_matrix_mpi::SeqProcForChecking(arr, base_input[2], ref_for_check);
    ASSERT_EQ(out, ref_for_check);
  }
}

TEST(veliev_e_sum_values_by_rows_matrix_mpi, Test_matr_1x1) {
  std::vector base_input = {1, 1, 1};

  // Create data
  boost::mpi::communicator world;
  std::vector<int> arr(base_input[0]);
  std::vector<int> out(base_input[1]);
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    veliev_e_sum_values_by_rows_matrix_mpi::GetRndMatrix(arr);
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(base_input.data()));  // num_elem + rows + cols
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(arr.data()));
    task_data->inputs_count.emplace_back(3);
    task_data->inputs_count.emplace_back(arr.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data->outputs_count.emplace_back(base_input[1]);
  }
  veliev_e_sum_values_by_rows_matrix_mpi::SumValuesByRowsMatrixMpi test1(task_data);
  ASSERT_EQ(test1.Validation(), true);
  test1.PreProcessing();
  test1.Run();
  test1.PostProcessing();
  if (world.rank() == 0) {
    std::vector<int> ref_for_check;
    veliev_e_sum_values_by_rows_matrix_mpi::SeqProcForChecking(arr, base_input[2], ref_for_check);
    ASSERT_EQ(out, ref_for_check);
  }
}

TEST(veliev_e_sum_values_by_rows_matrix_mpi, Test_matr_10x10) {
  boost::mpi::communicator world;

  std::vector base_input = {100, 10, 10};

  std::vector<int> arr(base_input[0]);
  std::vector<int> out(base_input[1]);
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    veliev_e_sum_values_by_rows_matrix_mpi::GetRndMatrix(arr);
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(base_input.data()));  // num_elem + rows + cols
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(arr.data()));
    task_data->inputs_count.emplace_back(3);
    task_data->inputs_count.emplace_back(arr.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data->outputs_count.emplace_back(base_input[1]);
  }
  veliev_e_sum_values_by_rows_matrix_mpi::SumValuesByRowsMatrixMpi test1(task_data);
  ASSERT_EQ(test1.Validation(), true);
  test1.PreProcessing();
  test1.Run();
  test1.PostProcessing();
  if (world.rank() == 0) {
    std::vector<int> ref_for_check;
    veliev_e_sum_values_by_rows_matrix_mpi::SeqProcForChecking(arr, base_input[2], ref_for_check);
    ASSERT_EQ(out, ref_for_check);
  }
}
TEST(veliev_e_sum_values_by_rows_matrix_mpi, Test_matr_11x11) {
  std::vector base_input = {121, 11, 11};

  // Create data
  boost::mpi::communicator world;
  std::vector<int> arr(base_input[0]);
  std::vector<int> out(base_input[1]);
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    veliev_e_sum_values_by_rows_matrix_mpi::GetRndMatrix(arr);
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(base_input.data()));  // num_elem + rows + cols
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(arr.data()));
    task_data->inputs_count.emplace_back(3);
    task_data->inputs_count.emplace_back(arr.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data->outputs_count.emplace_back(base_input[1]);
  }
  veliev_e_sum_values_by_rows_matrix_mpi::SumValuesByRowsMatrixMpi test1(task_data);
  ASSERT_EQ(test1.Validation(), true);
  test1.PreProcessing();
  test1.Run();
  test1.PostProcessing();
  if (world.rank() == 0) {
    std::vector<int> ref_for_check;
    veliev_e_sum_values_by_rows_matrix_mpi::SeqProcForChecking(arr, base_input[2], ref_for_check);
    ASSERT_EQ(out, ref_for_check);
  }
}

TEST(veliev_e_sum_values_by_rows_matrix_mpi, Test_matr_40x20) {
  std::vector base_input = {800, 40, 20};

  // Create data
  boost::mpi::communicator world;
  std::vector<int> arr(base_input[0]);
  std::vector<int> out(base_input[1]);
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    veliev_e_sum_values_by_rows_matrix_mpi::GetRndMatrix(arr);
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(base_input.data()));  // num_elem + rows + cols
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(arr.data()));
    task_data->inputs_count.emplace_back(3);
    task_data->inputs_count.emplace_back(arr.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data->outputs_count.emplace_back(base_input[1]);
  }
  veliev_e_sum_values_by_rows_matrix_mpi::SumValuesByRowsMatrixMpi test1(task_data);
  ASSERT_EQ(test1.Validation(), true);
  test1.PreProcessing();
  test1.Run();
  test1.PostProcessing();
  if (world.rank() == 0) {
    std::vector<int> ref_for_check;
    veliev_e_sum_values_by_rows_matrix_mpi::SeqProcForChecking(arr, base_input[2], ref_for_check);
    ASSERT_EQ(out, ref_for_check);
  }
}

TEST(veliev_e_sum_values_by_rows_matrix_mpi, Test_matr_100x100) {
  std::vector base_input = {10000, 100, 100};

  // Create data
  boost::mpi::communicator world;
  std::vector<int> arr(base_input[0]);
  std::vector<int> out(base_input[1]);
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    veliev_e_sum_values_by_rows_matrix_mpi::GetRndMatrix(arr);
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(base_input.data()));  // num_elem + rows + cols
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(arr.data()));
    task_data->inputs_count.emplace_back(3);
    task_data->inputs_count.emplace_back(arr.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data->outputs_count.emplace_back(base_input[1]);
  }
  veliev_e_sum_values_by_rows_matrix_mpi::SumValuesByRowsMatrixMpi test1(task_data);
  ASSERT_EQ(test1.Validation(), true);
  test1.PreProcessing();
  test1.Run();
  test1.PostProcessing();
  if (world.rank() == 0) {
    std::vector<int> ref_for_check;
    veliev_e_sum_values_by_rows_matrix_mpi::SeqProcForChecking(arr, base_input[2], ref_for_check);
    ASSERT_EQ(out, ref_for_check);
  }
}

TEST(veliev_e_sum_values_by_rows_matrix_mpi, Fixed_tests_1) {
  std::vector base_input = {9, 3, 3};

  // Create data
  boost::mpi::communicator world;
  std::vector<int> arr(base_input[0]);
  std::vector<int> out(base_input[1]);
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    arr = {1, 1, 1, 1, 1, 1, 1, 1, 1};
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(base_input.data()));  // num_elem + rows + cols
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(arr.data()));
    task_data->inputs_count.emplace_back(3);
    task_data->inputs_count.emplace_back(arr.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data->outputs_count.emplace_back(base_input[1]);
  }
  veliev_e_sum_values_by_rows_matrix_mpi::SumValuesByRowsMatrixMpi test1(task_data);
  ASSERT_EQ(test1.Validation(), true);
  test1.PreProcessing();
  test1.Run();
  test1.PostProcessing();
  if (world.rank() == 0) {
    std::vector<int> ref_for_check = {3, 3, 3};
    ASSERT_EQ(out, ref_for_check);
  }
}

TEST(veliev_e_sum_values_by_rows_matrix_mpi, Fixed_tests_2) {
  std::vector base_input = {4, 4, 1};

  // Create data
  boost::mpi::communicator world;
  std::vector<int> arr(base_input[0]);
  std::vector<int> out(base_input[1]);
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    arr = {4, 5, 6, 7};
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(base_input.data()));  // num_elem + rows + cols
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(arr.data()));
    task_data->inputs_count.emplace_back(3);
    task_data->inputs_count.emplace_back(arr.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data->outputs_count.emplace_back(base_input[1]);
  }
  veliev_e_sum_values_by_rows_matrix_mpi::SumValuesByRowsMatrixMpi test1(task_data);
  ASSERT_EQ(test1.Validation(), true);
  test1.PreProcessing();
  test1.Run();
  test1.PostProcessing();
  if (world.rank() == 0) {
    std::vector<int> ref_for_check = {4, 5, 6, 7};
    ASSERT_EQ(out, ref_for_check);
  }
}

TEST(veliev_e_sum_values_by_rows_matrix_mpi, Fixed_tests_3) {
  std::vector base_input = {16, 1, 16};

  // Create data
  boost::mpi::communicator world;
  std::vector<int> arr(base_input[0]);
  std::vector<int> out(base_input[1]);
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    arr = {
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    };
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(base_input.data()));  // num_elem + rows + cols
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(arr.data()));
    task_data->inputs_count.emplace_back(3);
    task_data->inputs_count.emplace_back(arr.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data->outputs_count.emplace_back(base_input[1]);
  }
  veliev_e_sum_values_by_rows_matrix_mpi::SumValuesByRowsMatrixMpi test1(task_data);
  ASSERT_EQ(test1.Validation(), true);
  test1.PreProcessing();
  test1.Run();
  test1.PostProcessing();
  if (world.rank() == 0) {
    std::vector<int> ref_for_check = {16};
    ASSERT_EQ(out, ref_for_check);
  }
}
