#include <gtest/gtest.h>

#include <algorithm>
#include <boost/mpi/communicator.hpp>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "mpi/somov_i_num_of_alternations_signs/include/num_of_alternations_signs_header_mpi_somov.hpp"

namespace {
void GetRndVector(std::vector<int> &vec) {
  std::random_device rd;
  std::default_random_engine reng(rd());
  std::uniform_int_distribution<int> dist(-static_cast<int>(vec.size()) - 1, static_cast<int>(vec.size()) - 1);
  std::ranges::generate(vec, [&dist, &reng] { return dist(reng); });
}
}  // namespace

TEST(somov_i_num_of_alternations_signs_mpi, Test_vec_0) {
  boost::mpi::communicator world;
  const int n = 0;
  // Create data
  std::vector<int> arr;
  int out = 0;
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    arr.resize(n);
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(arr.data()));
    task_data->inputs_count.emplace_back(arr.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out));
    task_data->outputs_count.emplace_back(1);
  }
  somov_i_num_of_alternations_signs_mpi::NumOfAlternationsSigns test1(task_data);
  if (world.rank() == 0) {
    ASSERT_EQ(test1.Validation(), false);
  }
}

TEST(somov_i_num_of_alternations_signs_mpi, Test_vec_1) {
  boost::mpi::communicator world;
  const int n = 1;
  // Create data
  std::vector<int> arr;
  int out = 0;
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    arr.resize(n);
    GetRndVector(arr);
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(arr.data()));
    task_data->inputs_count.emplace_back(arr.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out));
    task_data->outputs_count.emplace_back(1);
  }
  somov_i_num_of_alternations_signs_mpi::NumOfAlternationsSigns test1(task_data);
  ASSERT_EQ(test1.Validation(), true);
  test1.PreProcessing();
  test1.Run();
  test1.PostProcessing();
  if (world.rank() == 0) {
    int checker = 0;
    somov_i_num_of_alternations_signs_mpi::CheckForAlternationSigns(arr, checker);
    ASSERT_EQ(out, checker);
  }
}

TEST(somov_i_num_of_alternations_signs_mpi, Test_vec_999) {
  boost::mpi::communicator world;
  const int n = 999;
  // Create data
  std::vector<int> arr;
  int out = 0;
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    arr.resize(n);
    GetRndVector(arr);
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(arr.data()));
    task_data->inputs_count.emplace_back(arr.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out));
    task_data->outputs_count.emplace_back(1);
  }
  somov_i_num_of_alternations_signs_mpi::NumOfAlternationsSigns test1(task_data);

  ASSERT_EQ(test1.Validation(), true);
  test1.PreProcessing();
  test1.Run();
  test1.PostProcessing();
  if (world.rank() == 0) {
    int checker = 0;
    somov_i_num_of_alternations_signs_mpi::CheckForAlternationSigns(arr, checker);
    ASSERT_EQ(out, checker);
  }
}

TEST(somov_i_num_of_alternations_signs_mpi, Test_vec_10000) {
  boost::mpi::communicator world;
  const int n = 10000;
  // Create data
  std::vector<int> arr;
  int out = 0;
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    arr.resize(n);
    GetRndVector(arr);
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(arr.data()));
    task_data->inputs_count.emplace_back(arr.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out));
    task_data->outputs_count.emplace_back(1);
  }
  somov_i_num_of_alternations_signs_mpi::NumOfAlternationsSigns test1(task_data);

  ASSERT_EQ(test1.Validation(), true);
  test1.PreProcessing();
  test1.Run();
  test1.PostProcessing();
  if (world.rank() == 0) {
    int checker = 0;
    somov_i_num_of_alternations_signs_mpi::CheckForAlternationSigns(arr, checker);
    ASSERT_EQ(out, checker);
  }
}

TEST(somov_i_num_of_alternations_signs_mpi, Test_vec_731) {
  boost::mpi::communicator world;
  const int n = 731;
  // Create data
  std::vector<int> arr;
  int out = 0;
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    arr.resize(n);
    GetRndVector(arr);
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(arr.data()));
    task_data->inputs_count.emplace_back(arr.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out));
    task_data->outputs_count.emplace_back(1);
  }
  somov_i_num_of_alternations_signs_mpi::NumOfAlternationsSigns test1(task_data);

  ASSERT_EQ(test1.Validation(), true);
  test1.PreProcessing();
  test1.Run();
  test1.PostProcessing();
  if (world.rank() == 0) {
    int checker = 0;
    somov_i_num_of_alternations_signs_mpi::CheckForAlternationSigns(arr, checker);
    ASSERT_EQ(out, checker);
  }
}
TEST(somov_i_num_of_alternations_signs_mpi, Norandom_check_1) {
  boost::mpi::communicator world;
  // Create data
  std::vector<int> arr;
  int out = 0;
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    arr = {-1, 1, -1, 1};
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(arr.data()));
    task_data->inputs_count.emplace_back(arr.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out));
    task_data->outputs_count.emplace_back(1);
  }
  somov_i_num_of_alternations_signs_mpi::NumOfAlternationsSigns test1(task_data);
  ASSERT_EQ(test1.Validation(), true);
  test1.PreProcessing();
  test1.Run();
  test1.PostProcessing();
  if (world.rank() == 0) {
    ASSERT_EQ(out, 3);
  }
}

TEST(somov_i_num_of_alternations_signs_mpi, Norandom_check_2) {
  boost::mpi::communicator world;
  // Create data
  std::vector<int> arr;
  int out = 0;
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    arr = {-1, 1, -1, 1, 1, 1, 1, 1, 1, 1};
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(arr.data()));
    task_data->inputs_count.emplace_back(arr.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out));
    task_data->outputs_count.emplace_back(1);
  }
  somov_i_num_of_alternations_signs_mpi::NumOfAlternationsSigns test1(task_data);
  ASSERT_EQ(test1.Validation(), true);
  test1.PreProcessing();
  test1.Run();
  test1.PostProcessing();
  if (world.rank() == 0) {
    ASSERT_EQ(out, 3);
  }
}

TEST(somov_i_num_of_alternations_signs_mpi, Norandom_check_3) {
  boost::mpi::communicator world;
  // Create data
  std::vector<int> arr;
  int out = 0;
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    arr = {-1, 1, -1, 1, 1, 1, -1, 1, -1, 1};
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(arr.data()));
    task_data->inputs_count.emplace_back(arr.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out));
    task_data->outputs_count.emplace_back(1);
  }
  somov_i_num_of_alternations_signs_mpi::NumOfAlternationsSigns test1(task_data);
  ASSERT_EQ(test1.Validation(), true);

  test1.PreProcessing();
  test1.Run();
  test1.PostProcessing();
  if (world.rank() == 0) {
    ASSERT_EQ(out, 7);
  }
}