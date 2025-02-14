// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <string>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/budazhapova_e_count_freq_character/include/count_freq_character_mpi_header.hpp"

namespace budazhapova_e_count_freq_character_mpi {
std::string getRandomString(int length) {
  static std::string charset = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890";
  std::string result;
  result.resize(length);

  srand(time(nullptr));
  for (int i = 0; i < length; i++) result[i] = charset[rand() % charset.length()];

  return result;
}
}  // namespace budazhapova_e_count_freq_character_mpi

TEST(budazhapova_e_count_freq_character_mpi, test_pipeline_run) {
  boost::mpi::communicator world;
  std::string global_str;
  const int size_string = 123456789;
  global_str = budazhapova_e_count_freq_character_mpi::getRandomString(size_string);
  std::vector<int> global_out(1, 0);
  char symb = 'a';
  // Create TaskData

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_str.data()));
    taskDataPar->inputs_count.emplace_back(global_str.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&symb));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_out.data()));
    taskDataPar->outputs_count.emplace_back(global_out.size());
  }
  auto testMpiTaskParallel = std::make_shared<budazhapova_e_count_freq_character_mpi::TestMPITaskParallel>(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel->Validation(), true);
  testMpiTaskParallel->PreProcessing();
  testMpiTaskParallel->Run();
  testMpiTaskParallel->PostProcessing();
  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };
  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->PipelineRun(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perfResults);
    // ASSERT_EQ(, global_out[0]);
  }
}
TEST(budazhapova_e_count_freq_character_mpi, test_task_run) {
  boost::mpi::communicator world;
  std::string global_str;
  const int size_string = 123456789;
  global_str = budazhapova_e_count_freq_character_mpi::getRandomString(size_string);
  std::vector<int> global_out(1, 0);
  char symb = 'a';

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_str.data()));
    taskDataPar->inputs_count.emplace_back(global_str.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&symb));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_out.data()));
    taskDataPar->outputs_count.emplace_back(global_out.size());
  }
  auto testMpiTaskParallel = std::make_shared<budazhapova_e_count_freq_character_mpi::TestMPITaskParallel>(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel->Validation(), true);
  testMpiTaskParallel->PreProcessing();
  testMpiTaskParallel->Run();
  testMpiTaskParallel->PostProcessing();
  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };
  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->TaskRun(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perfResults);
    ASSERT_EQ(123456789, global_out[0]);
  }
}