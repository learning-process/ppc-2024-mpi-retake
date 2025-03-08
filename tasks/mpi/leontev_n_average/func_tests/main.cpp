// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <cstdint>
#include <memory>
#include <numeric>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "mpi/leontev_n_average/include/ops_mpi.hpp"

namespace {
std::vector<int> GetRandomVector(int sz) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vec(sz);
  for (int i = 0; i < sz; i++) {
    vec[i] = static_cast<int>(gen()) % 100;
  }
  return vec;
}

inline void TaskEmplacement(std::shared_ptr<ppc::core::TaskData>& task_data_par, std::vector<int>& global_vec,
                            std::vector<int32_t>& global_avg) {
  task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
  task_data_par->inputs_count.emplace_back(global_vec.size());
  task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_avg.data()));
  task_data_par->outputs_count.emplace_back(global_avg.size());
}
}  // namespace

TEST(leontev_n_average_mpi, avg_mpi_50elem) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int32_t> global_avg(1, 0);
  int32_t expected_avg = 0;
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    const int vector_size = 50;
    global_vec = GetRandomVector(vector_size);
    TaskEmplacement(task_data_par, global_vec, global_avg);
  }
  leontev_n_average_mpi::MPIVecAvgParallel mpi_vec_avg_parallel(task_data_par);
  ASSERT_TRUE(mpi_vec_avg_parallel.Validation());
  mpi_vec_avg_parallel.PreProcessing();
  mpi_vec_avg_parallel.Run();
  mpi_vec_avg_parallel.PostProcessing();
  if (world.rank() == 0) {
    expected_avg = std::accumulate(global_vec.begin(), global_vec.end(), 0) / static_cast<int>(global_vec.size());
    ASSERT_EQ(expected_avg, global_avg[0]);
  }
}
TEST(leontev_n_average_mpi, avg_mpi_0elem) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int32_t> global_avg(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    TaskEmplacement(task_data_par, global_vec, global_avg);
  }
  leontev_n_average_mpi::MPIVecAvgParallel mpi_vec_avg_parallel(task_data_par);
  if (world.rank() == 0) {
    ASSERT_FALSE(mpi_vec_avg_parallel.Validation());
  }
}
TEST(leontev_n_average_mpi, avg_mpi_1000elem) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int32_t> global_avg(1, 0);
  int32_t expected_avg = 0;
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    const int vector_size = 1000;
    global_vec = GetRandomVector(vector_size);
    TaskEmplacement(task_data_par, global_vec, global_avg);
  }
  leontev_n_average_mpi::MPIVecAvgParallel mpi_vec_avg_parallel(task_data_par);
  ASSERT_TRUE(mpi_vec_avg_parallel.Validation());
  mpi_vec_avg_parallel.PreProcessing();
  mpi_vec_avg_parallel.Run();
  mpi_vec_avg_parallel.PostProcessing();
  if (world.rank() == 0) {
    expected_avg = std::accumulate(global_vec.begin(), global_vec.end(), 0) / static_cast<int>(global_vec.size());
    ASSERT_EQ(expected_avg, global_avg[0]);
  }
}
TEST(leontev_n_average_mpi, avg_mpi_20000elem) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int32_t> global_avg(1, 0);
  int32_t expected_avg = 0;
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    const int vector_size = 20000;
    global_vec = GetRandomVector(vector_size);
    TaskEmplacement(task_data_par, global_vec, global_avg);
  }
  leontev_n_average_mpi::MPIVecAvgParallel mpi_vec_avg_parallel(task_data_par);
  ASSERT_TRUE(mpi_vec_avg_parallel.Validation());
  mpi_vec_avg_parallel.PreProcessing();
  mpi_vec_avg_parallel.Run();
  mpi_vec_avg_parallel.PostProcessing();
  if (world.rank() == 0) {
    expected_avg = std::accumulate(global_vec.begin(), global_vec.end(), 0) / static_cast<int>(global_vec.size());
    ASSERT_EQ(expected_avg, global_avg[0]);
  }
}
TEST(leontev_n_average_mpi, avg_mpi_1elem) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int32_t> global_avg(1, 0);
  int32_t expected_avg = 0;
  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    const int vector_size = 1;
    global_vec = GetRandomVector(vector_size);
    TaskEmplacement(task_data_par, global_vec, global_avg);
  }
  leontev_n_average_mpi::MPIVecAvgParallel mpi_vec_avg_parallel(task_data_par);
  ASSERT_TRUE(mpi_vec_avg_parallel.Validation());
  mpi_vec_avg_parallel.PreProcessing();
  mpi_vec_avg_parallel.Run();
  mpi_vec_avg_parallel.PostProcessing();
  if (world.rank() == 0) {
    expected_avg = std::accumulate(global_vec.begin(), global_vec.end(), 0) / static_cast<int>(global_vec.size());
    ASSERT_EQ(expected_avg, global_avg[0]);
  }
}
