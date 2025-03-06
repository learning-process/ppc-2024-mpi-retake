#include <gtest/gtest.h>

#include "mpi/fomin_v_sentence_count/include/ops_mpi.hpp"

TEST(fomin_v_sentence_count, Test_Empty_String) {
  boost::mpi::communicator world;
  std::string input = "";
  int result = 0;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> task_data_mpi = std::make_shared<ppc::core::TaskData>();
  task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<char*>(input.data())));
  task_data_mpi->inputs_count.emplace_back(input.size());
  task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result));
  task_data_mpi->outputs_count.emplace_back(1);

  // Create and run parallel task
  fomin_v_sentence_count::SentenceCountParallel sentenceCountParallel(task_data_mpi);
  ASSERT_EQ(sentenceCountParallel.validation(), true);
  sentenceCountParallel.PreProcessingImpl();
  sentenceCountParallel.RunImpl();
  sentenceCountParallel.PostProcessingImpl();

  if (world.rank() == 0) {
    ASSERT_EQ(result, 0);
  }
}

TEST(fomin_v_sentence_count, Test_Single_Sentence) {
  boost::mpi::communicator world;
  std::string input = "Hello world.";
  int result = 0;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> task_data_mpi = std::make_shared<ppc::core::TaskData>();
  task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<char*>(input.data())));
  task_data_mpi->inputs_count.emplace_back(input.size());
  task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result));
  task_data_mpi->outputs_count.emplace_back(1);

  // Create and run parallel task
  fomin_v_sentence_count::SentenceCountParallel sentenceCountParallel(task_data_mpi);
  ASSERT_EQ(sentenceCountParallel.validation(), true);
  sentenceCountParallel.PreProcessingImpl();
  sentenceCountParallel.RunImpl();
  sentenceCountParallel.PostProcessingImpl();

  if (world.rank() == 0) {
    ASSERT_EQ(result, 1);
  }
}

TEST(fomin_v_sentence_count, Test_Multiple_Sentences) {
  boost::mpi::communicator world;
  std::string input = "Hello world. How are you? I'm fine!";
  int result = 0;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> task_data_mpi = std::make_shared<ppc::core::TaskData>();
  task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<char*>(input.data())));
  task_data_mpi->inputs_count.emplace_back(input.size());
  task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result));
  task_data_mpi->outputs_count.emplace_back(1);

  // Create and run parallel task
  fomin_v_sentence_count::SentenceCountParallel sentenceCountParallel(task_data_mpi);
  ASSERT_EQ(sentenceCountParallel.validation(), true);
  sentenceCountParallel.PreProcessingImpl();
  sentenceCountParallel.RunImpl();
  sentenceCountParallel.PostProcessingImpl();

  if (world.rank() == 0) {
    ASSERT_EQ(result, 3);
  }
}

TEST(fomin_v_sentence_count, Test_Sequential_Consistency) {
  boost::mpi::communicator world;
  std::string input = "This is a test. Another test! And one more?";
  int parallel_result = 0;
  int sequential_result = 0;

  // Create TaskData for parallel task
  std::shared_ptr<ppc::core::TaskData> task_data_mpi = std::make_shared<ppc::core::TaskData>();
  task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<char*>(input.data())));
  task_data_mpi->inputs_count.emplace_back(input.size());
  task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(&parallel_result));
  task_data_mpi->outputs_count.emplace_back(1);

  // Create and run parallel task
  fomin_v_sentence_count::SentenceCountParallel sentenceCountParallel(task_data_mpi);
  ASSERT_EQ(sentenceCountParallel.ValidationImpl(), true);
  sentenceCountParallel.PreProcessingImpl();
  sentenceCountParallel.RunImpl();
  sentenceCountParallel.PostProcessingImpl();

  if (world.rank() == 0) {
    // Create TaskData for sequential task
    std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<char*>(input.data())));
    task_data_seq->inputs_count.emplace_back(input.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(&sequential_result));
    task_data_seq->outputs_count.emplace_back(1);

    // Create and run sequential task
    fomin_v_sentence_count::SentenceCountSequential sentenceCountSequential(task_data_seq);
    ASSERT_EQ(sentenceCountSequential.ValidationImpl(), true);
    sentenceCountSequential.PreProcessingImpl();
    sentenceCountSequential.RunImpl();
    sentenceCountSequential.PostProcessingImpl();

    // Compare results
    ASSERT_EQ(parallel_result, sequential_result);
  }
}