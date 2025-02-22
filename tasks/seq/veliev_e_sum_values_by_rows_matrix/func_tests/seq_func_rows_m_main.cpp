#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/veliev_e_sum_values_by_rows_matrix/include/seq_rows_m_header.hpp"

TEST(veliev_e_sum_values_by_rows_matrix_seq, Test_matr_0x0) {
  std::vector base_input = {0, 0, 0};

  // Create data
  std::vector<int> arr(base_input[0]);
  veliev_e_sum_values_by_rows_matrix_seq::GetRndMatrix(arr);
  std::vector<int> out(base_input[1]);
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(base_input.data()));  // num_elem + rows + cols
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(arr.data()));
  task_data->inputs_count.emplace_back(3);
  task_data->inputs_count.emplace_back(arr.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data->outputs_count.emplace_back(base_input[1]);

  veliev_e_sum_values_by_rows_matrix_seq::SumValuesByRowsMatrixSeq test1(task_data);

  ASSERT_EQ(test1.Validation(), true);
  test1.PreProcessing();
  test1.Run();
  test1.PostProcessing();
  std::vector<int> ref_for_check;
  veliev_e_sum_values_by_rows_matrix_seq::SeqProcForChecking(arr, base_input[2], ref_for_check);
  ASSERT_EQ(out, ref_for_check);
}

TEST(veliev_e_sum_values_by_rows_matrix_seq, Test_matr_1x1) {
  std::vector base_input = {1, 1, 1};

  // Create data
  std::vector<int> arr(base_input[0]);
  veliev_e_sum_values_by_rows_matrix_seq::GetRndMatrix(arr);
  std::vector<int> out(base_input[1]);
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(base_input.data()));  // num_elem + rows + cols
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(arr.data()));
  task_data->inputs_count.emplace_back(3);
  task_data->inputs_count.emplace_back(arr.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data->outputs_count.emplace_back(base_input[1]);

  veliev_e_sum_values_by_rows_matrix_seq::SumValuesByRowsMatrixSeq test1(task_data);

  ASSERT_EQ(test1.Validation(), true);
  test1.PreProcessing();
  test1.Run();
  test1.PostProcessing();
  std::vector<int> ref_for_check;
  veliev_e_sum_values_by_rows_matrix_seq::SeqProcForChecking(arr, base_input[2], ref_for_check);
  ASSERT_EQ(out, ref_for_check);
}

TEST(veliev_e_sum_values_by_rows_matrix_seq, Test_matr_10x10) {
  std::vector base_input = {100, 10, 10};

  // Create data
  std::vector<int> arr(base_input[0]);
  veliev_e_sum_values_by_rows_matrix_seq::GetRndMatrix(arr);
  std::vector<int> out(base_input[1]);
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(base_input.data()));  // num_elem + rows + cols
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(arr.data()));
  task_data->inputs_count.emplace_back(3);
  task_data->inputs_count.emplace_back(arr.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data->outputs_count.emplace_back(base_input[1]);

  veliev_e_sum_values_by_rows_matrix_seq::SumValuesByRowsMatrixSeq test1(task_data);

  ASSERT_EQ(test1.Validation(), true);
  test1.PreProcessing();
  test1.Run();
  test1.PostProcessing();
  std::vector<int> ref_for_check;
  veliev_e_sum_values_by_rows_matrix_seq::SeqProcForChecking(arr, base_input[2], ref_for_check);
  ASSERT_EQ(out, ref_for_check);
}

TEST(veliev_e_sum_values_by_rows_matrix_seq, Test_matr_40x60) {
  std::vector base_input = {2400, 40, 60};

  // Create data
  std::vector<int> arr(base_input[0]);
  veliev_e_sum_values_by_rows_matrix_seq::GetRndMatrix(arr);
  std::vector<int> out(base_input[1]);
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(base_input.data()));  // num_elem + rows + cols
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(arr.data()));
  task_data->inputs_count.emplace_back(3);
  task_data->inputs_count.emplace_back(arr.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data->outputs_count.emplace_back(base_input[1]);

  veliev_e_sum_values_by_rows_matrix_seq::SumValuesByRowsMatrixSeq test1(task_data);

  ASSERT_EQ(test1.Validation(), true);
  test1.PreProcessing();
  test1.Run();
  test1.PostProcessing();
  std::vector<int> ref_for_check;
  veliev_e_sum_values_by_rows_matrix_seq::SeqProcForChecking(arr, base_input[2], ref_for_check);
  ASSERT_EQ(out, ref_for_check);
}

TEST(veliev_e_sum_values_by_rows_matrix_seq, Test_matr_100x100) {
  std::vector base_input = {10000, 100, 100};

  // Create data
  std::vector<int> arr(base_input[0]);
  veliev_e_sum_values_by_rows_matrix_seq::GetRndMatrix(arr);
  std::vector<int> out(base_input[1]);
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(base_input.data()));  // num_elem + rows + cols
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(arr.data()));
  task_data->inputs_count.emplace_back(3);
  task_data->inputs_count.emplace_back(arr.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data->outputs_count.emplace_back(base_input[1]);

  veliev_e_sum_values_by_rows_matrix_seq::SumValuesByRowsMatrixSeq test1(task_data);

  ASSERT_EQ(test1.Validation(), true);
  test1.PreProcessing();
  test1.Run();
  test1.PostProcessing();
  std::vector<int> ref_for_check;
  veliev_e_sum_values_by_rows_matrix_seq::SeqProcForChecking(arr, base_input[2], ref_for_check);
  ASSERT_EQ(out, ref_for_check);
}

TEST(veliev_e_sum_values_by_rows_matrix_seq, Fixed_tests_1) {
  std::vector base_input = {9, 3, 3};

  // Create data
  std::vector<int> arr(base_input[0]);
  arr = {2, 2, 2, 2, 2, 2, 2, 2, 2};
  std::vector<int> out(base_input[1]);
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(base_input.data()));  // num_elem + rows + cols
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(arr.data()));
  task_data->inputs_count.emplace_back(3);
  task_data->inputs_count.emplace_back(arr.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data->outputs_count.emplace_back(base_input[1]);

  veliev_e_sum_values_by_rows_matrix_seq::SumValuesByRowsMatrixSeq test1(task_data);

  ASSERT_EQ(test1.Validation(), true);
  test1.PreProcessing();
  test1.Run();
  test1.PostProcessing();
  std::vector<int> ref_for_check = {6, 6, 6};
  ASSERT_EQ(out, ref_for_check);
}

TEST(veliev_e_sum_values_by_rows_matrix_seq, Fixed_tests_2) {
  std::vector base_input = {16, 1, 16};

  // Create data
  std::vector<int> arr(base_input[0]);
  arr = {
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
  };
  std::vector<int> out(base_input[1]);
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(base_input.data()));  // num_elem + rows + cols
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(arr.data()));
  task_data->inputs_count.emplace_back(3);
  task_data->inputs_count.emplace_back(arr.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data->outputs_count.emplace_back(base_input[1]);

  veliev_e_sum_values_by_rows_matrix_seq::SumValuesByRowsMatrixSeq test1(task_data);

  ASSERT_EQ(test1.Validation(), true);
  test1.PreProcessing();
  test1.Run();
  test1.PostProcessing();
  std::vector<int> ref_for_check = {16};
  ASSERT_EQ(out, ref_for_check);
}

TEST(veliev_e_sum_values_by_rows_matrix_seq, Fixed_tests_3) {
  std::vector base_input = {3, 3, 1};

  // Create data
  std::vector<int> arr(base_input[0]);
  arr = {7, 7, 7};
  std::vector<int> out(base_input[1]);
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(base_input.data()));  // num_elem + rows + cols
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(arr.data()));
  task_data->inputs_count.emplace_back(3);
  task_data->inputs_count.emplace_back(arr.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data->outputs_count.emplace_back(base_input[1]);

  veliev_e_sum_values_by_rows_matrix_seq::SumValuesByRowsMatrixSeq test1(task_data);

  ASSERT_EQ(test1.Validation(), true);
  test1.PreProcessing();
  test1.Run();
  test1.PostProcessing();
  std::vector<int> ref_for_check = {7, 7, 7};
  ASSERT_EQ(out, ref_for_check);
}