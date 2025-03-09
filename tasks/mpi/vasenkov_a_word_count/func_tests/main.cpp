#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"
#include "mpi/vasenkov_a_word_count/include/ops_mpi.hpp"

TEST(vasenkov_a_word_count_mpi, test_0_word) {
  boost::mpi::communicator world;
  std::string input;
  input = "";
  std::vector<uint8_t> in(input.begin(), input.end());
  std::vector<int> out(1, 0);
  std::vector<int> expect = {0};
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  vasenkov_a_word_count_mpi::WordCountMPI test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();

  if (world.rank() == 0) {
    EXPECT_EQ(out, expect);
  }
}

TEST(vasenkov_a_word_count_mpi, test_1_word) {
  boost::mpi::communicator world;
  std::string input = "Hello.";
  std::vector<uint8_t> in(input.begin(), input.end());
  std::vector<int> out(1, 0);
  std::vector<int> expect = {1};
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  vasenkov_a_word_count_mpi::WordCountMPI test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();

  if (world.rank() == 0) {
    EXPECT_EQ(out, expect);
  }
}
TEST(vasenkov_a_word_count_mpi, test_2_word) {
  boost::mpi::communicator world;
  std::string input = "Hello. world!";
  std::vector<uint8_t> in(input.begin(), input.end());
  std::vector<int> out(1, 0);
  std::vector<int> expect = {2};
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  vasenkov_a_word_count_mpi::WordCountMPI test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();

  if (world.rank() == 0) {
    EXPECT_EQ(out, expect);
  }
}

TEST(vasenkov_a_word_count_mpi, test_3_word) {
  boost::mpi::communicator world;
  std::string input = "Hello. world! World";
  std::vector<uint8_t> in(input.begin(), input.end());
  std::vector<int> out(1, 0);
  std::vector<int> expect = {3};
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  vasenkov_a_word_count_mpi::WordCountMPI test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();

  if (world.rank() == 0) {
    EXPECT_EQ(out, expect);
  }
}

TEST(vasenkov_a_word_count_mpi, test_4_word) {
  boost::mpi::communicator world;
  std::string input = "Hello. world! a`m this";
  std::vector<uint8_t> in(input.begin(), input.end());
  std::vector<int> out(1, 0);
  std::vector<int> expect = {4};
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  vasenkov_a_word_count_mpi::WordCountMPI test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();

  if (world.rank() == 0) {
    EXPECT_EQ(out, expect);
  }
}
TEST(vasenkov_a_word_count_mpi, test_1_word_and_spacees) {
  boost::mpi::communicator world;
  std::string input = " Hello. ";
  std::vector<uint8_t> in(input.begin(), input.end());
  std::vector<int> out(1, 0);
  std::vector<int> expect = {1};
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  vasenkov_a_word_count_mpi::WordCountMPI test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();

  if (world.rank() == 0) {
    EXPECT_EQ(out, expect);
  }
}

TEST(vasenkov_a_word_count_mpi, test_2_space_begin) {
  boost::mpi::communicator world;
  std::string input = "  Hello. world!";
  std::vector<uint8_t> in(input.begin(), input.end());
  std::vector<int> out(1, 0);
  std::vector<int> expect = {2};
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  vasenkov_a_word_count_mpi::WordCountMPI test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  if (world.rank() == 0) {
    EXPECT_EQ(out, expect);
  }
}

TEST(vasenkov_a_word_count_mpi, test_2_space_end) {
  boost::mpi::communicator world;
  std::string input = "Hello. world!  ";
  std::vector<uint8_t> in(input.begin(), input.end());
  std::vector<int> out(1, 0);
  std::vector<int> expect = {2};
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  vasenkov_a_word_count_mpi::WordCountMPI test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  if (world.rank() == 0) {
    EXPECT_EQ(out, expect);
  }
}

TEST(vasenkov_a_word_count_mpi, test_2_space_middle) {
  boost::mpi::communicator world;
  std::string input = "Hello.  world!";
  std::vector<uint8_t> in(input.begin(), input.end());
  std::vector<int> out(1, 0);
  std::vector<int> expect = {2};
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  vasenkov_a_word_count_mpi::WordCountMPI test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  if (world.rank() == 0) {
    EXPECT_EQ(out, expect);
  }
} /*
 TEST(vasenkov_a_word_count_mpi, test_more_space_middle) {
   std::string input = "Hello.        world!";
   boost::mpi::communicator world;

   std::vector<uint8_t> in(input.begin(), input.end());
   std::vector<int> out(1, 0);
   std::vector<int> expect = {2};
   auto task_data_seq = std::make_shared<ppc::core::TaskData>();
   if (world.rank() == 0) {
     task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
     task_data_seq->inputs_count.emplace_back(in.size());
     task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
     task_data_seq->outputs_count.emplace_back(out.size());
   }
   vasenkov_a_word_count_mpi::WordCountMPI test_task_sequential(task_data_seq);
   ASSERT_EQ(test_task_sequential.Validation(), true);
   test_task_sequential.PreProcessing();
   test_task_sequential.Run();
   test_task_sequential.PostProcessing();
   if (world.rank() == 0) {
     EXPECT_EQ(out, expect);
   }
 }*/
