#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <memory>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"
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

TEST(budazhapova_e_count_freq_character_mpi, test_with_random_string) {
  boost::mpi::communicator world;
  std::string global_str;
  std::vector<int> global_out(1, 0);
  char symb = '1';
  const int size_string = 10;
  global_str = budazhapova_e_count_freq_character_mpi::getRandomString(size_string);

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
  budazhapova_e_count_freq_character_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.Validation(), true);
  testMpiTaskParallel.PreProcessing();
  testMpiTaskParallel.Run();
  testMpiTaskParallel.PostProcessing();
  if (world.rank() == 0) {
    // Create data
    std::vector<int> reference_out(1, 0);
    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_str.data()));
    taskDataSeq->inputs_count.emplace_back(global_str.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&symb));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_out.data()));
    taskDataSeq->outputs_count.emplace_back(reference_out.size());

    // Create Task
    budazhapova_e_count_freq_character_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.Validation(), true);
    testMpiTaskSequential.PreProcessing();
    testMpiTaskSequential.Run();
    testMpiTaskSequential.Post_processing();
    ASSERT_EQ(reference_out[0], global_out[0]);
  }
}
TEST(budazhapova_e_count_freq_character_mpi, test_with_one_symb) {
  boost::mpi::communicator world;
  std::string global_str = "3";
  std::vector<int> global_out(1, 0);
  char symb = '3';
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
  budazhapova_e_count_freq_character_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.Validation(), true);
  testMpiTaskParallel.PreProcessing();
  testMpiTaskParallel.Run();
  testMpiTaskParallel.PostProcessing();
  if (world.rank() == 0) {
    // Create data
    std::vector<int> reference_out(1, 0);
    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_str.data()));
    taskDataSeq->inputs_count.emplace_back(global_str.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&symb));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_out.data()));
    taskDataSeq->outputs_count.emplace_back(reference_out.size());

    // Create Task
    budazhapova_e_count_freq_character_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.Validation(), true);
    testMpiTaskSequential.PreProcessing();
    testMpiTaskSequential.Run();
    testMpiTaskSequential.PostProcessing();
    ASSERT_EQ(reference_out[0], global_out[0]);
  }
}
TEST(budazhapova_e_count_freq_character_mpi, big_string) {
  boost::mpi::communicator world;
  std::string global_str(100, 'a');
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
  budazhapova_e_count_freq_character_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.Validation(), true);
  testMpiTaskParallel.PreProcessing();
  testMpiTaskParallel.Run();
  testMpiTaskParallel.PostProcessing();
  if (world.rank() == 0) {
    // Create data
    std::vector<int> reference_out(1, 0);
    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_str.data()));
    taskDataSeq->inputs_count.emplace_back(global_str.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&symb));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_out.data()));
    taskDataSeq->outputs_count.emplace_back(reference_out.size());

    // Create Task
    budazhapova_e_count_freq_character_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.Validation(), true);
    testMpiTaskSequential.PreProcessing();
    testMpiTaskSequential.Run();
    testMpiTaskSequential.PostProcessing();
    ASSERT_EQ(reference_out[0], global_out[0]);
  }
}