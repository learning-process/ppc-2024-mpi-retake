#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>
#include "mpi/Shpynov_N_reader_writer/include/readers_writers_mpi.hpp"

TEST(Shpynov_N_readers_writers, test_single) {
    boost::mpi::communicator world;
  
    if (world.size() < 2) {
      GTEST_SKIP();
    }
  
    std::vector<int> crit_res (1,0);

    int expected_result = 0;
    std::vector<int> returned_result (1,0);
    std::shared_ptr<ppc::core::TaskData> task_data_mpi = std::make_shared<ppc::core::TaskData>();

    if (world.rank() == 0) {

      int writers_count = 0;
      for (int i = 1; i < world.size(); i++) {
        if (i % 2 != 0) writers_count++;
      }
      expected_result = writers_count;

      task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(crit_res.data()));
      task_data_mpi->inputs_count.emplace_back(1);
      task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(returned_result.data()));
      task_data_mpi->outputs_count.emplace_back(1);
    }

    Shpynov_N_readers_writers_mpi::TestTaskMPI TestTaskMPI(task_data_mpi);
    ASSERT_EQ(TestTaskMPI.ValidationImpl(), true);
    TestTaskMPI.PreProcessingImpl();
    TestTaskMPI.RunImpl();
    TestTaskMPI.PostProcessingImpl();
    if (world.rank() == 0) {
      ASSERT_EQ(expected_result, returned_result[0]);
    }
  }

TEST(Shpynov_N_readers_writers, test_alot) {
    boost::mpi::communicator world;

    if (world.size() < 2) {
      GTEST_SKIP();
    }

    std::vector<int> crit_res = {1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3};

    std::vector<int> expected_result(crit_res.size());
    std::vector<int> returned_result(crit_res.size());

    std::shared_ptr<ppc::core::TaskData> task_data_mpi = std::make_shared<ppc::core::TaskData>();

    if (world.rank() == 0) {
      int writers_count = 0;
      for (int i = 1; i < world.size(); i++) {
        if (i % 2 != 0) writers_count++;
      }
      for (int i = 0; i < crit_res.size(); i++) {
        expected_result[i] = writers_count+crit_res[i];
      }
      task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(crit_res.data()));
      task_data_mpi->inputs_count.emplace_back(crit_res.size());
      task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(returned_result.data()));
      task_data_mpi->outputs_count.emplace_back(crit_res.size());
    }

    Shpynov_N_readers_writers_mpi::TestTaskMPI TestTaskMPI(task_data_mpi);
    ASSERT_EQ(TestTaskMPI.ValidationImpl(), true);
    TestTaskMPI.PreProcessingImpl();
    TestTaskMPI.RunImpl();
    TestTaskMPI.PostProcessingImpl();
    if (world.rank() == 0) {
      ASSERT_EQ(expected_result, returned_result);
    }
  }

TEST(Shpynov_N_readers_writers, test_multiple) {
    boost::mpi::communicator world;

    if (world.size() < 2) {
      GTEST_SKIP();
    }

    std::vector<int> crit_res = {1, 2, 3};

    std::vector<int> expected_result(crit_res.size());
    std::vector<int> returned_result(crit_res.size());

    std::shared_ptr<ppc::core::TaskData> task_data_mpi = std::make_shared<ppc::core::TaskData>();

    if (world.rank() == 0) {
      int writers_count = 0;
      for (int i = 1; i < world.size(); i++) {
        if (i % 2 != 0) writers_count++;
      }
      for (int i = 0; i < crit_res.size(); i++) {
        expected_result[i] = writers_count + crit_res[i];
      }
      task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(crit_res.data()));
      task_data_mpi->inputs_count.emplace_back(crit_res.size());
      task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(returned_result.data()));
      task_data_mpi->outputs_count.emplace_back(crit_res.size());
    }

    Shpynov_N_readers_writers_mpi::TestTaskMPI TestTaskMPI(task_data_mpi);
    ASSERT_EQ(TestTaskMPI.ValidationImpl(), true);
    TestTaskMPI.PreProcessingImpl();
    TestTaskMPI.RunImpl();
    TestTaskMPI.PostProcessingImpl();
    if (world.rank() == 0) {
      ASSERT_EQ(expected_result, returned_result);
    }
  }

TEST(Shpynov_N_readers_writers, test_invalid_size) {
    boost::mpi::communicator world;

    if (world.size() < 2) {
      GTEST_SKIP();
    }

    std::vector<int> crit_res = {};

    std::vector<int> expected_result(crit_res.size());
    std::vector<int> returned_result(crit_res.size());

    std::shared_ptr<ppc::core::TaskData> task_data_mpi = std::make_shared<ppc::core::TaskData>();

    if (world.rank() == 0) {
      int writers_count = 0;
      for (int i = 1; i < world.size(); i++) {
        if (i % 2 != 0) writers_count++;
      }
      for (int i = 0; i < crit_res.size(); i++) {
        expected_result[i] = writers_count + crit_res[i];
      }
      task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(crit_res.data()));
      task_data_mpi->inputs_count.emplace_back(crit_res.size());
      task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(returned_result.data()));
      task_data_mpi->outputs_count.emplace_back(crit_res.size());
    }

    Shpynov_N_readers_writers_mpi::TestTaskMPI TestTaskMPI(task_data_mpi);

    if (world.rank() == 0) {
      ASSERT_NE(TestTaskMPI.ValidationImpl(), true);
    }
  }

TEST(Shpynov_N_readers_writers, test_different_sizes) {
    boost::mpi::communicator world;

    if (world.size() < 2) {
      GTEST_SKIP();
    }

    std::vector<int> crit_res = {};

    std::vector<int> expected_result(crit_res.size()+1);
    std::vector<int> returned_result(crit_res.size());

    std::shared_ptr<ppc::core::TaskData> task_data_mpi = std::make_shared<ppc::core::TaskData>();

    if (world.rank() == 0) {
      int writers_count = 0;
      for (int i = 1; i < world.size(); i++) {
        if (i % 2 != 0) writers_count++;
      }
      for (int i = 0; i < crit_res.size(); i++) {
        expected_result[i] = writers_count + crit_res[i];
      }
      task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(crit_res.data()));
      task_data_mpi->inputs_count.emplace_back(crit_res.size());
      task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(returned_result.data()));
      task_data_mpi->outputs_count.emplace_back(crit_res.size());
    }

    Shpynov_N_readers_writers_mpi::TestTaskMPI TestTaskMPI(task_data_mpi);

    if (world.rank() == 0) {
      ASSERT_NE(TestTaskMPI.ValidationImpl(), true);
    }
  }