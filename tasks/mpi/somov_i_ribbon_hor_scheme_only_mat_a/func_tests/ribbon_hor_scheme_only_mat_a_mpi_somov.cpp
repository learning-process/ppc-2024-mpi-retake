#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "mpi/somov_i_ribbon_hor_scheme_only_mat_a/include/somov_i_ribbon_hor_scheme_only_mat_a_mpi.hpp"
TEST(somov_i_ribbon_hor_scheme_only_mat_a_mpi, Test_3x3_fixed) {
  boost::mpi::communicator world;
  const int a_c = 2;
  const int a_r = 2;
  const int b_c = 2;
  const int b_r = 2;
  std::vector<int> a;
  std::vector<int> b;
  std::vector<int> c;
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    a.resize(a_c * a_r);
    b.resize(b_r * b_c);
    c.resize(a_r * b_c);
    a = {1, 2, 3, 4};
    b = {5, 6, 7, 8};
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    task_data->inputs_count.emplace_back(a_c);
    task_data->inputs_count.emplace_back(a_r);
    task_data->inputs_count.emplace_back(b_c);
    task_data->inputs_count.emplace_back(b_r);
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(c.data()));
    task_data->outputs_count.emplace_back(c.size());
  }
  somov_i_ribbon_hor_scheme_only_mat_a_mpi::RibbonHorSchemeOnlyMatA test1(task_data);
  ASSERT_EQ(test1.Validation(), true);
  test1.PreProcessing();
  test1.Run();
  test1.PostProcessing();
  if (world.rank() == 0) {
    std::vector<int> checker(a_r * b_c, 0);
    somov_i_ribbon_hor_scheme_only_mat_a_mpi::LiterallyMult(a, b, checker, a_c, a_r, b_c);
    ASSERT_EQ(checker, c);
  }
}

TEST(somov_i_ribbon_hor_scheme_only_mat_a_mpi, Test_3x3) {
  boost::mpi::communicator world;
  const int a_c = 3;
  const int a_r = 3;
  const int b_c = 3;
  const int b_r = 3;
  std::vector<int> a;
  std::vector<int> b;
  std::vector<int> c;
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    a.resize(a_c * a_r);
    b.resize(b_r * b_c);
    c.resize(a_r * b_c);
    somov_i_ribbon_hor_scheme_only_mat_a_mpi::GetRndVector(a);
    somov_i_ribbon_hor_scheme_only_mat_a_mpi::GetRndVector(b);
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    task_data->inputs_count.emplace_back(a_c);
    task_data->inputs_count.emplace_back(a_r);
    task_data->inputs_count.emplace_back(b_c);
    task_data->inputs_count.emplace_back(b_r);
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(c.data()));
    task_data->outputs_count.emplace_back(c.size());
  }
  somov_i_ribbon_hor_scheme_only_mat_a_mpi::RibbonHorSchemeOnlyMatA test1(task_data);
  ASSERT_EQ(test1.Validation(), true);
  test1.PreProcessing();
  test1.Run();
  test1.PostProcessing();
  if (world.rank() == 0) {
    std::vector<int> checker(a_r * b_c, 0);
    somov_i_ribbon_hor_scheme_only_mat_a_mpi::LiterallyMult(a, b, checker, a_c, a_r, b_c);
    ASSERT_EQ(checker, c);
  }
}
TEST(somov_i_ribbon_hor_scheme_only_mat_a_mpi, Test_10x10) {
  boost::mpi::communicator world;
  const int a_c = 10;
  const int a_r = 10;
  const int b_c = 10;
  const int b_r = 10;
  std::vector<int> a;
  std::vector<int> b;
  std::vector<int> c;
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    a.resize(a_c * a_r);
    b.resize(b_r * b_c);
    c.resize(a_r * b_c);
    somov_i_ribbon_hor_scheme_only_mat_a_mpi::GetRndVector(a);
    somov_i_ribbon_hor_scheme_only_mat_a_mpi::GetRndVector(b);
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    task_data->inputs_count.emplace_back(a_c);
    task_data->inputs_count.emplace_back(a_r);
    task_data->inputs_count.emplace_back(b_c);
    task_data->inputs_count.emplace_back(b_r);
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(c.data()));
    task_data->outputs_count.emplace_back(c.size());
  }
  somov_i_ribbon_hor_scheme_only_mat_a_mpi::RibbonHorSchemeOnlyMatA test1(task_data);
  ASSERT_EQ(test1.Validation(), true);
  test1.PreProcessing();
  test1.Run();
  test1.PostProcessing();
  if (world.rank() == 0) {
    std::vector<int> checker(a_r * b_c, 0);
    somov_i_ribbon_hor_scheme_only_mat_a_mpi::LiterallyMult(a, b, checker, a_c, a_r, b_c);
    ASSERT_EQ(checker, c);
  }
}
TEST(somov_i_ribbon_hor_scheme_only_mat_a_mpi, Test_100x100) {
  boost::mpi::communicator world;
  const int a_c = 100;
  const int a_r = 100;
  const int b_c = 100;
  const int b_r = 100;
  std::vector<int> a;
  std::vector<int> b;
  std::vector<int> c;
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    a.resize(a_c * a_r);
    b.resize(b_r * b_c);
    c.resize(a_r * b_c);
    somov_i_ribbon_hor_scheme_only_mat_a_mpi::GetRndVector(a);
    somov_i_ribbon_hor_scheme_only_mat_a_mpi::GetRndVector(b);
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    task_data->inputs_count.emplace_back(a_c);
    task_data->inputs_count.emplace_back(a_r);
    task_data->inputs_count.emplace_back(b_c);
    task_data->inputs_count.emplace_back(b_r);
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(c.data()));
    task_data->outputs_count.emplace_back(c.size());
  }
  somov_i_ribbon_hor_scheme_only_mat_a_mpi::RibbonHorSchemeOnlyMatA test1(task_data);
  ASSERT_EQ(test1.Validation(), true);
  test1.PreProcessing();
  test1.Run();
  test1.PostProcessing();
  if (world.rank() == 0) {
    std::vector<int> checker(a_r * b_c, 0);
    somov_i_ribbon_hor_scheme_only_mat_a_mpi::LiterallyMult(a, b, checker, a_c, a_r, b_c);
    ASSERT_EQ(checker, c);
  }
}
TEST(somov_i_ribbon_hor_scheme_only_mat_a_mpi, Test_17x17) {
  boost::mpi::communicator world;
  const int a_c = 17;
  const int a_r = 17;
  const int b_c = 17;
  const int b_r = 17;
  std::vector<int> a;
  std::vector<int> b;
  std::vector<int> c;
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    a.resize(a_c * a_r);
    b.resize(b_r * b_c);
    c.resize(a_r * b_c);
    somov_i_ribbon_hor_scheme_only_mat_a_mpi::GetRndVector(a);
    somov_i_ribbon_hor_scheme_only_mat_a_mpi::GetRndVector(b);
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    task_data->inputs_count.emplace_back(a_c);
    task_data->inputs_count.emplace_back(a_r);
    task_data->inputs_count.emplace_back(b_c);
    task_data->inputs_count.emplace_back(b_r);
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(c.data()));
    task_data->outputs_count.emplace_back(c.size());
  }
  somov_i_ribbon_hor_scheme_only_mat_a_mpi::RibbonHorSchemeOnlyMatA test1(task_data);
  ASSERT_EQ(test1.Validation(), true);
  test1.PreProcessing();
  test1.Run();
  test1.PostProcessing();
  if (world.rank() == 0) {
    std::vector<int> checker(a_r * b_c, 0);
    somov_i_ribbon_hor_scheme_only_mat_a_mpi::LiterallyMult(a, b, checker, a_c, a_r, b_c);
    ASSERT_EQ(checker, c);
  }
}
TEST(somov_i_ribbon_hor_scheme_only_mat_a_mpi, Non_square_matrix_1) {
  boost::mpi::communicator world;
  const int a_r = 17;
  const int a_c = 18;
  const int b_r = 18;
  const int b_c = 19;
  std::vector<int> a;
  std::vector<int> b;
  std::vector<int> c;
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    a.resize(a_c * a_r);
    b.resize(b_r * b_c);
    c.resize(a_r * b_c);
    somov_i_ribbon_hor_scheme_only_mat_a_mpi::GetRndVector(a);
    somov_i_ribbon_hor_scheme_only_mat_a_mpi::GetRndVector(b);
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    task_data->inputs_count.emplace_back(a_c);
    task_data->inputs_count.emplace_back(a_r);
    task_data->inputs_count.emplace_back(b_c);
    task_data->inputs_count.emplace_back(b_r);
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(c.data()));
    task_data->outputs_count.emplace_back(c.size());
  }
  somov_i_ribbon_hor_scheme_only_mat_a_mpi::RibbonHorSchemeOnlyMatA test1(task_data);
  ASSERT_EQ(test1.Validation(), true);
  test1.PreProcessing();
  test1.Run();
  test1.PostProcessing();
  if (world.rank() == 0) {
    std::vector<int> checker(a_r * b_c, 0);
    somov_i_ribbon_hor_scheme_only_mat_a_mpi::LiterallyMult(a, b, checker, a_c, a_r, b_c);
    ASSERT_EQ(checker, c);
  }
}
TEST(somov_i_ribbon_hor_scheme_only_mat_a_mpi, Non_square_matrix_2) {
  boost::mpi::communicator world;
  const int a_r = 2;
  const int a_c = 1;
  const int b_r = 1;
  const int b_c = 2;
  std::vector<int> a;
  std::vector<int> b;
  std::vector<int> c;
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    a = {1, 1};
    b = {2, 2};
    c.resize(4);
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    task_data->inputs_count.emplace_back(a_c);
    task_data->inputs_count.emplace_back(a_r);
    task_data->inputs_count.emplace_back(b_c);
    task_data->inputs_count.emplace_back(b_r);
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(c.data()));
    task_data->outputs_count.emplace_back(c.size());
  }
  somov_i_ribbon_hor_scheme_only_mat_a_mpi::RibbonHorSchemeOnlyMatA test1(task_data);
  ASSERT_EQ(test1.Validation(), true);
  test1.PreProcessing();
  test1.Run();
  test1.PostProcessing();
  if (world.rank() == 0) {
    std::vector<int> checker(a_r * b_c, 0);
    somov_i_ribbon_hor_scheme_only_mat_a_mpi::LiterallyMult(a, b, checker, a_c, a_r, b_c);
    ASSERT_EQ(checker, c);
  }
  ASSERT_TRUE(true);
}
TEST(somov_i_ribbon_hor_scheme_only_mat_a_mpi, Non_square_matrix_3) {
  boost::mpi::communicator world;
  const int a_r = 1;
  const int a_c = 3;
  const int b_r = 3;
  const int b_c = 1;
  std::vector<int> a;
  std::vector<int> b;
  std::vector<int> c;
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    a.resize(a_c * a_r);
    b.resize(b_r * b_c);
    c.resize(a_r * b_c);
    somov_i_ribbon_hor_scheme_only_mat_a_mpi::GetRndVector(a);
    somov_i_ribbon_hor_scheme_only_mat_a_mpi::GetRndVector(b);
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    task_data->inputs_count.emplace_back(a_c);
    task_data->inputs_count.emplace_back(a_r);
    task_data->inputs_count.emplace_back(b_c);
    task_data->inputs_count.emplace_back(b_r);
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(c.data()));
    task_data->outputs_count.emplace_back(c.size());
  }
  somov_i_ribbon_hor_scheme_only_mat_a_mpi::RibbonHorSchemeOnlyMatA test1(task_data);
  ASSERT_EQ(test1.Validation(), true);
  test1.PreProcessing();
  test1.Run();
  test1.PostProcessing();
  if (world.rank() == 0) {
    std::vector<int> checker(a_r * b_c, 0);
    somov_i_ribbon_hor_scheme_only_mat_a_mpi::LiterallyMult(a, b, checker, a_c, a_r, b_c);
    ASSERT_EQ(checker, c);
  }
}
