#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/somov_i_num_of_alternations_signs/include/num_of_alternations_signs_header_seq_somov.hpp"

TEST(somov_i_num_of_alternations_signs_seq, Test_vec_0) {
  const int n = 0;
  // Create data
  std::vector<int> arr(n);
  int out = 0;
  somov_i_num_of_alternations_signs_seq::GetRndVector(arr);
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(arr.data()));
  task_data->inputs_count.emplace_back(arr.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out));
  task_data->outputs_count.emplace_back(1);

  somov_i_num_of_alternations_signs_seq::NumOfAlternationsSigns test1(task_data);

  ASSERT_EQ(test1.Validation(), true);
  test1.PreProcessing();
  test1.Run();
  test1.PostProcessing();
  int checker = 0;
  somov_i_num_of_alternations_signs_seq::CheckForAlternationSigns(arr, checker);
  ASSERT_EQ(out, checker);
}

TEST(somov_i_num_of_alternations_signs_seq, Test_vec_1) {
  const int n = 1;
  // Create data
  std::vector<int> arr(n);
  int out = 0;
  somov_i_num_of_alternations_signs_seq::GetRndVector(arr);
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(arr.data()));
  task_data->inputs_count.emplace_back(arr.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out));
  task_data->outputs_count.emplace_back(1);

  somov_i_num_of_alternations_signs_seq::NumOfAlternationsSigns test1(task_data);

  ASSERT_EQ(test1.Validation(), true);
  test1.PreProcessing();
  test1.Run();
  test1.PostProcessing();
  int checker = 0;
  somov_i_num_of_alternations_signs_seq::CheckForAlternationSigns(arr, checker);
  ASSERT_EQ(out, checker);
}

TEST(somov_i_num_of_alternations_signs_seq, Test_vec_1000) {
  const int n = 1000;
  // Create data
  std::vector<int> arr(n);
  int out = 0;
  somov_i_num_of_alternations_signs_seq::GetRndVector(arr);
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(arr.data()));
  task_data->inputs_count.emplace_back(arr.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out));
  task_data->outputs_count.emplace_back(1);

  somov_i_num_of_alternations_signs_seq::NumOfAlternationsSigns test1(task_data);

  ASSERT_EQ(test1.Validation(), true);
  test1.PreProcessing();
  test1.Run();
  test1.PostProcessing();
  int checker = 0;
  somov_i_num_of_alternations_signs_seq::CheckForAlternationSigns(arr, checker);
  ASSERT_EQ(out, checker);
}

TEST(somov_i_num_of_alternations_signs_seq, Test_vec_10000) {
  const int n = 10000;
  // Create data
  std::vector<int> arr(n);
  int out = 0;
  somov_i_num_of_alternations_signs_seq::GetRndVector(arr);
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(arr.data()));
  task_data->inputs_count.emplace_back(arr.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out));
  task_data->outputs_count.emplace_back(1);

  somov_i_num_of_alternations_signs_seq::NumOfAlternationsSigns test1(task_data);

  ASSERT_EQ(test1.Validation(), true);
  test1.PreProcessing();
  test1.Run();
  test1.PostProcessing();
  int checker = 0;
  somov_i_num_of_alternations_signs_seq::CheckForAlternationSigns(arr, checker);
  ASSERT_EQ(out, checker);
}

TEST(somov_i_num_of_alternations_signs_seq, Test_vec_731) {
  const int n = 731;
  // Create data
  std::vector<int> arr(n);
  int out = 0;
  somov_i_num_of_alternations_signs_seq::GetRndVector(arr);
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(arr.data()));
  task_data->inputs_count.emplace_back(arr.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out));
  task_data->outputs_count.emplace_back(1);

  somov_i_num_of_alternations_signs_seq::NumOfAlternationsSigns test1(task_data);

  ASSERT_EQ(test1.Validation(), true);
  test1.PreProcessing();
  test1.Run();
  test1.PostProcessing();
  int checker = 0;
  somov_i_num_of_alternations_signs_seq::CheckForAlternationSigns(arr, checker);
  ASSERT_EQ(out, checker);
}
TEST(somov_i_num_of_alternations_signs_seq, Norandom_check_1) {
  const int n = 3;
  // Create data
  std::vector<int> arr(n);
  int out = 0;
  arr = {-1, 1, -1};
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(arr.data()));
  task_data->inputs_count.emplace_back(arr.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out));
  task_data->outputs_count.emplace_back(1);

  somov_i_num_of_alternations_signs_seq::NumOfAlternationsSigns test1(task_data);

  ASSERT_EQ(test1.Validation(), true);
  test1.PreProcessing();
  test1.Run();
  test1.PostProcessing();
  ASSERT_EQ(out, 2);
}

TEST(somov_i_num_of_alternations_signs_seq, Norandom_check_2) {
  const int n = 4;
  // Create data
  std::vector<int> arr(n);
  int out = 0;
  arr = {-1, -1, 1, 1};
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(arr.data()));
  task_data->inputs_count.emplace_back(arr.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out));
  task_data->outputs_count.emplace_back(1);

  somov_i_num_of_alternations_signs_seq::NumOfAlternationsSigns test1(task_data);

  ASSERT_EQ(test1.Validation(), true);
  test1.PreProcessing();
  test1.Run();
  test1.PostProcessing();
  ASSERT_EQ(out, 1);
}

TEST(somov_i_num_of_alternations_signs_seq, Norandom_check_3) {
  const int n = 4;
  // Create data
  std::vector<int> arr(n);
  int out = 0;
  arr = {-1, 1, -1, 1};
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(arr.data()));
  task_data->inputs_count.emplace_back(arr.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out));
  task_data->outputs_count.emplace_back(1);

  somov_i_num_of_alternations_signs_seq::NumOfAlternationsSigns test1(task_data);

  ASSERT_EQ(test1.Validation(), true);
  test1.PreProcessing();
  test1.Run();
  test1.PostProcessing();
  ASSERT_EQ(out, 3);
}