#include <gtest/gtest.h>

#include <cstdlib>
#include <ctime>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "core/util/include/util.hpp"
#include "mpi/budazhapova_betcher_odd_even_merge_mpi/include/radix_sort_with_betcher.h"

namespace budazhapova_betcher_odd_even_merge_mpi {
namespace {
std::vector<int> GenerateRandomVector(int size, int min_value, int max_value) {
  std::vector<int> random_vector(size);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(min_value, max_value);
  for (int i = 0; i < size; ++i) {
    random_vector[i] = dis(gen);
  }

  return random_vector;
}
}  // namespace
}  // namespace budazhapova_betcher_odd_even_merge_mpi

TEST(budazhapova_betcher_odd_even_merge_mpi, ordinary_test) {
  boost::mpi::communicator world;
  std::vector<int> input_vector = {34, 12, 5, 78, 23, 45, 67, 89, 10, 2, 56, 43};
  std::vector<int> out(12, 0);
  std::vector<int> out_seq(12, 0);

  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_vector.data()));
    task_data_par->inputs_count.emplace_back(input_vector.size());
    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  }
  budazhapova_betcher_odd_even_merge_mpi::MergeParallel test_mpi_task_parallel(task_data_par);
  ASSERT_EQ(test_mpi_task_parallel.Validation(), true);
  test_mpi_task_parallel.PreProcessing();
  test_mpi_task_parallel.Run();
  test_mpi_task_parallel.PostProcessing();

  if (world.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_vector.data()));
    task_data_seq->inputs_count.emplace_back(input_vector.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_seq.data()));

    budazhapova_betcher_odd_even_merge_mpi::MergeSequential testTaskSequential(task_data_seq);
    ASSERT_EQ(testTaskSequential.Validation(), true);
    testTaskSequential.PreProcessing();
    testTaskSequential.Run();
    testTaskSequential.PostProcessing();

    ASSERT_EQ(out, out_seq);
  }
}

TEST(budazhapova_betcher_odd_even_merge_mpi, random_vector_test) {
  boost::mpi::communicator world;
  std::vector<int> input_vector = budazhapova_betcher_odd_even_merge_mpi::GenerateRandomVector(120, 5, 100);
  std::vector<int> out(120, 0);
  std::vector<int> out_seq(120, 0);

  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_vector.data()));
    task_data_par->inputs_count.emplace_back(input_vector.size());
    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  }
  budazhapova_betcher_odd_even_merge_mpi::MergeParallel test_mpi_task_parallel(task_data_par);
  ASSERT_EQ(test_mpi_task_parallel.Validation(), true);
  test_mpi_task_parallel.PreProcessing();
  test_mpi_task_parallel.Run();
  test_mpi_task_parallel.PostProcessing();

  if (world.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_vector.data()));
    task_data_seq->inputs_count.emplace_back(input_vector.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_seq.data()));

    budazhapova_betcher_odd_even_merge_mpi::MergeSequential testTaskSequential(task_data_seq);
    ASSERT_EQ(testTaskSequential.Validation(), true);
    testTaskSequential.PreProcessing();
    testTaskSequential.Run();
    testTaskSequential.PostProcessing();

    ASSERT_EQ(out, out_seq);
  }
}

TEST(budazhapova_betcher_odd_even_merge_mpi, random_vector_test_2) {
  boost::mpi::communicator world;
  std::vector<int> input_vector = budazhapova_betcher_odd_even_merge_mpi::GenerateRandomVector(85, 5, 100);
  std::vector<int> out(85, 0);
  std::vector<int> out_seq(85, 0);

  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_vector.data()));
    task_data_par->inputs_count.emplace_back(input_vector.size());
    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  }
  budazhapova_betcher_odd_even_merge_mpi::MergeParallel test_mpi_task_parallel(task_data_par);
  ASSERT_EQ(test_mpi_task_parallel.Validation(), true);
  test_mpi_task_parallel.PreProcessing();
  test_mpi_task_parallel.Run();
  test_mpi_task_parallel.PostProcessing();

  if (world.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_vector.data()));
    task_data_seq->inputs_count.emplace_back(input_vector.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_seq.data()));

    budazhapova_betcher_odd_even_merge_mpi::MergeSequential testTaskSequential(task_data_seq);
    ASSERT_EQ(testTaskSequential.Validation(), true);
    testTaskSequential.PreProcessing();
    testTaskSequential.Run();
    testTaskSequential.PostProcessing();

    ASSERT_EQ(out, out_seq);
  }
}

TEST(budazhapova_betcher_odd_even_merge_mpi, random_vector_test_3) {
  boost::mpi::communicator world;
  std::vector<int> input_vector = budazhapova_betcher_odd_even_merge_mpi::GenerateRandomVector(100, 5, 100);
  std::vector<int> out(100, 0);
  std::vector<int> out_seq(100, 0);

  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_vector.data()));
    task_data_par->inputs_count.emplace_back(input_vector.size());
    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  }
  budazhapova_betcher_odd_even_merge_mpi::MergeParallel test_mpi_task_parallel(task_data_par);
  ASSERT_EQ(test_mpi_task_parallel.Validation(), true);
  test_mpi_task_parallel.PreProcessing();
  test_mpi_task_parallel.Run();
  test_mpi_task_parallel.PostProcessing();

  if (world.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_vector.data()));
    task_data_seq->inputs_count.emplace_back(input_vector.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_seq.data()));

    budazhapova_betcher_odd_even_merge_mpi::MergeSequential testTaskSequential(task_data_seq);
    ASSERT_EQ(testTaskSequential.Validation(), true);
    testTaskSequential.PreProcessing();
    testTaskSequential.Run();
    testTaskSequential.PostProcessing();

    ASSERT_EQ(out, out_seq);
  }
}

TEST(budazhapova_betcher_odd_even_merge_mpi, validation_test) {
  boost::mpi::communicator world;
  std::vector<int> input_vector = {};
  std::vector<int> out(1, 0);

  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_vector.data()));
    task_data_par->inputs_count.emplace_back(input_vector.size());
    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    task_data_par->outputs_count.emplace_back(out.size());
    budazhapova_betcher_odd_even_merge_mpi::MergeParallel test_mpi_task_parallel(task_data_par);
    ASSERT_EQ(test_mpi_task_parallel.Validation(), false);
  }
}