#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "boost/mpi/communicator.hpp"
#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "mpi/fomin_v_sentence_count/include/ops_mpi.hpp"

TEST(fomin_v_sentence_count, test_parallel_pipeline_run) {
  boost::mpi::communicator world;
  std::string global_text;
  std::vector<int32_t> global_sentence_count(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> task_data_mpi = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_text = "Hello! How are you? I am fine.";
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_text.data()));
    task_data_mpi->inputs_count.emplace_back(global_text.size());
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_sentence_count.data()));
    task_data_mpi->outputs_count.emplace_back(global_sentence_count.size());
  }

  auto sentenceCountParallel = std::make_shared<fomin_v_sentence_count::SentenceCountParallel>(task_data_mpi);
  ASSERT_EQ(sentenceCountParallel->ValidationImpl(), true);
  sentenceCountParallel->PreProcessingImpl();
  sentenceCountParallel->RunImpl();
  sentenceCountParallel->PostProcessingImpl();

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(sentenceCountParallel);
  perfAnalyzer->PipelineRun(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perfResults);
    ASSERT_EQ(3, global_sentence_count[0]);  // Ожидаемое количество предложений
  }
}

TEST(fomin_v_sentence_count, test_sequential_task_run) {
  boost::mpi::communicator world;
  std::string global_text;
  std::vector<int32_t> global_sentence_count(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_text = "Hello! How are you? I am fine.";
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_text.data()));
    task_data_seq->inputs_count.emplace_back(global_text.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_sentence_count.data()));
    task_data_seq->outputs_count.emplace_back(global_sentence_count.size());
  }

  auto sentenceCountSequential = std::make_shared<fomin_v_sentence_count::SentenceCountSequential>(task_data_seq);
  ASSERT_EQ(sentenceCountSequential->ValidationImpl(), true);
  sentenceCountSequential->PreProcessingImpl();
  sentenceCountSequential->RunImpl();
  sentenceCountSequential->PostProcessingImpl();

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(sentenceCountSequential);
  perfAnalyzer->TaskRun(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perfResults);
    ASSERT_EQ(3, global_sentence_count[0]);  // Ожидаемое количество предложений
  }
}