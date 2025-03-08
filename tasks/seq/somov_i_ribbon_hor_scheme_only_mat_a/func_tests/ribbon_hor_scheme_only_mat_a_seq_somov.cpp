#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/somov_i_ribbon_hor_scheme_only_mat_a/include/ribbon_hor_scheme_only_mat_a_header_seq_somov.hpp"

TEST(somov_i_ribbon_hor_scheme_only_mat_a_seq, Test_3x3_matirx) {
  // Create data
  const int a_c = 3;
  const int a_r = 3;
  const int b_c = 3;
  const int b_r = 3;
  std::vector<int> a(a_c * a_r);
  std::vector<int> b(b_r * b_c);
  std::vector<int> c(a_r * b_c);
  somov_i_ribbon_hor_scheme_only_mat_a_seq::GetRndVector(a);
  somov_i_ribbon_hor_scheme_only_mat_a_seq::GetRndVector(b);
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  task_data->inputs_count.emplace_back(a_c);
  task_data->inputs_count.emplace_back(a_r);
  task_data->inputs_count.emplace_back(b_c);
  task_data->inputs_count.emplace_back(b_r);
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(c.data()));
  task_data->outputs_count.emplace_back(c.size());

  somov_i_ribbon_hor_scheme_only_mat_a_seq::RibbonHorSchemeOnlyMatA test1(task_data);

  ASSERT_EQ(test1.Validation(), true);
  test1.PreProcessing();
  test1.Run();
  test1.PostProcessing();
  std::vector<int> checker(a_r * b_c, 0);
  somov_i_ribbon_hor_scheme_only_mat_a_seq::LiterallyMult(a, b, checker, a_c, a_r, b_c);
  ASSERT_EQ(checker, c);
}

TEST(somov_i_ribbon_hor_scheme_only_mat_a_seq, Test_10x10_matirx) {
  // Create data
  const int a_r = 10;
  const int a_c = 10;
  const int b_r = 10;
  const int b_c = 10;

  std::vector<int> a(a_c * a_r);
  std::vector<int> b(b_r * b_c);
  std::vector<int> c(a_r * b_c);
  somov_i_ribbon_hor_scheme_only_mat_a_seq::GetRndVector(a);
  somov_i_ribbon_hor_scheme_only_mat_a_seq::GetRndVector(b);
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  task_data->inputs_count.emplace_back(a_c);
  task_data->inputs_count.emplace_back(a_r);
  task_data->inputs_count.emplace_back(b_c);
  task_data->inputs_count.emplace_back(b_r);
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(c.data()));
  task_data->outputs_count.emplace_back(c.size());

  somov_i_ribbon_hor_scheme_only_mat_a_seq::RibbonHorSchemeOnlyMatA test1(task_data);

  ASSERT_EQ(test1.Validation(), true);
  test1.PreProcessing();
  test1.Run();
  test1.PostProcessing();
  std::vector<int> checker(a_r * b_c, 0);
  somov_i_ribbon_hor_scheme_only_mat_a_seq::LiterallyMult(a, b, checker, a_c, a_r, b_c);
  ASSERT_EQ(checker, c);
}

TEST(somov_i_ribbon_hor_scheme_only_mat_a_seq, Test_17x17_matirx) {
  // Create data
  const int a_c = 17;
  const int a_r = 17;
  const int b_c = 17;
  const int b_r = 17;
  std::vector<int> a(a_c * a_r);
  std::vector<int> b(b_r * b_c);
  std::vector<int> c(a_r * b_c);
  somov_i_ribbon_hor_scheme_only_mat_a_seq::GetRndVector(a);
  somov_i_ribbon_hor_scheme_only_mat_a_seq::GetRndVector(b);
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  task_data->inputs_count.emplace_back(a_c);
  task_data->inputs_count.emplace_back(a_r);
  task_data->inputs_count.emplace_back(b_c);
  task_data->inputs_count.emplace_back(b_r);
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(c.data()));
  task_data->outputs_count.emplace_back(c.size());

  somov_i_ribbon_hor_scheme_only_mat_a_seq::RibbonHorSchemeOnlyMatA test1(task_data);

  ASSERT_EQ(test1.Validation(), true);
  test1.PreProcessing();
  test1.Run();
  test1.PostProcessing();
  std::vector<int> checker(a_r * b_c, 0);
  somov_i_ribbon_hor_scheme_only_mat_a_seq::LiterallyMult(a, b, checker, a_c, a_r, b_c);
  ASSERT_EQ(checker, c);
}

TEST(somov_i_ribbon_hor_scheme_only_mat_a_seq, Test_100x100_matirx) {
  // Create data
  const int a_c = 100;
  const int a_r = 100;
  const int b_c = 100;
  const int b_r = 100;
  std::vector<int> a(a_c * a_r);
  std::vector<int> b(b_r * b_c);
  std::vector<int> c(a_r * b_c);
  somov_i_ribbon_hor_scheme_only_mat_a_seq::GetRndVector(a);
  somov_i_ribbon_hor_scheme_only_mat_a_seq::GetRndVector(b);
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  task_data->inputs_count.emplace_back(a_c);
  task_data->inputs_count.emplace_back(a_r);
  task_data->inputs_count.emplace_back(b_c);
  task_data->inputs_count.emplace_back(b_r);
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(c.data()));
  task_data->outputs_count.emplace_back(c.size());

  somov_i_ribbon_hor_scheme_only_mat_a_seq::RibbonHorSchemeOnlyMatA test1(task_data);

  ASSERT_EQ(test1.Validation(), true);
  test1.PreProcessing();
  test1.Run();
  test1.PostProcessing();
  std::vector<int> checker(a_r * b_c, 0);
  somov_i_ribbon_hor_scheme_only_mat_a_seq::LiterallyMult(a, b, checker, a_c, a_r, b_c);
  ASSERT_EQ(checker, c);
}

TEST(somov_i_ribbon_hor_scheme_only_mat_a_seq, Non_square_matrix_1) {
  // Create data
  const int a_r = 17;
  const int a_c = 18;
  const int b_r = 18;
  const int b_c = 19;

  std::vector<int> a(a_c * a_r);
  std::vector<int> b(b_r * b_c);
  std::vector<int> c(a_r * b_c);
  somov_i_ribbon_hor_scheme_only_mat_a_seq::GetRndVector(a);
  somov_i_ribbon_hor_scheme_only_mat_a_seq::GetRndVector(b);
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  task_data->inputs_count.emplace_back(a_c);
  task_data->inputs_count.emplace_back(a_r);
  task_data->inputs_count.emplace_back(b_c);
  task_data->inputs_count.emplace_back(b_r);
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(c.data()));
  task_data->outputs_count.emplace_back(c.size());

  somov_i_ribbon_hor_scheme_only_mat_a_seq::RibbonHorSchemeOnlyMatA test1(task_data);

  ASSERT_EQ(test1.Validation(), true);
  test1.PreProcessing();
  test1.Run();
  test1.PostProcessing();
  std::vector<int> checker(a_r * b_c, 0);
  somov_i_ribbon_hor_scheme_only_mat_a_seq::LiterallyMult(a, b, checker, a_c, a_r, b_c);
  ASSERT_EQ(checker, c);
}

TEST(somov_i_ribbon_hor_scheme_only_mat_a_seq, Non_square_matrix_2) {
  // Create data
  const int a_r = 3;
  const int a_c = 1;
  const int b_r = 1;
  const int b_c = 2;
  std::vector<int> a(a_c * a_r);
  std::vector<int> b(b_r * b_c);
  std::vector<int> c(a_r * b_c);
  somov_i_ribbon_hor_scheme_only_mat_a_seq::GetRndVector(a);
  somov_i_ribbon_hor_scheme_only_mat_a_seq::GetRndVector(b);
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  task_data->inputs_count.emplace_back(a_c);
  task_data->inputs_count.emplace_back(a_r);
  task_data->inputs_count.emplace_back(b_c);
  task_data->inputs_count.emplace_back(b_r);
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(c.data()));
  task_data->outputs_count.emplace_back(c.size());

  somov_i_ribbon_hor_scheme_only_mat_a_seq::RibbonHorSchemeOnlyMatA test1(task_data);

  ASSERT_EQ(test1.Validation(), true);
  test1.PreProcessing();
  test1.Run();
  test1.PostProcessing();
  std::vector<int> checker(a_r * b_c, 0);
  somov_i_ribbon_hor_scheme_only_mat_a_seq::LiterallyMult(a, b, checker, a_c, a_r, b_c);
  ASSERT_EQ(checker, c);
}

TEST(somov_i_ribbon_hor_scheme_only_mat_a_seq, Non_square_matrix_3) {
  // Create data
  const int a_r = 1;
  const int a_c = 3;
  const int b_r = 3;
  const int b_c = 1;
  std::vector<int> a(a_c * a_r);
  std::vector<int> b(b_r * b_c);
  std::vector<int> c(a_r * b_c);
  somov_i_ribbon_hor_scheme_only_mat_a_seq::GetRndVector(a);
  somov_i_ribbon_hor_scheme_only_mat_a_seq::GetRndVector(b);
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  task_data->inputs_count.emplace_back(a_c);
  task_data->inputs_count.emplace_back(a_r);
  task_data->inputs_count.emplace_back(b_c);
  task_data->inputs_count.emplace_back(b_r);
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(c.data()));
  task_data->outputs_count.emplace_back(c.size());

  somov_i_ribbon_hor_scheme_only_mat_a_seq::RibbonHorSchemeOnlyMatA test1(task_data);

  ASSERT_EQ(test1.Validation(), true);
  test1.PreProcessing();
  test1.Run();
  test1.PostProcessing();
  std::vector<int> checker(a_r * b_c, 0);
  somov_i_ribbon_hor_scheme_only_mat_a_seq::LiterallyMult(a, b, checker, a_c, a_r, b_c);
  ASSERT_EQ(checker, c);
}