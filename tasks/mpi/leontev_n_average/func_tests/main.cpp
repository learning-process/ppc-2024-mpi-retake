// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <random>
#include <vector>

#include "mpi/leontev_n_average/include/ops_mpi.hpp"

namespace leontev_n_average_mpi {
std::vector<int> getRandomVector(int sz) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vec(sz);
  for (int i = 0; i < sz; i++) {
    vec[i] = gen() % 100;
  }
  return vec;
}
}  // namespace leontev_n_average_mpi

inline void taskEmplacement(std::shared_ptr<ppc::core::TaskData>& taskDataPar, std::vector<int>& global_vec,
                            std::vector<int32_t>& global_avg) {
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
  taskDataPar->inputs_count.emplace_back(global_vec.size());
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_avg.data()));
  taskDataPar->outputs_count.emplace_back(global_avg.size());
}

TEST(leontev_n_average_mpi, avg_mpi_50elem) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int32_t> global_avg(1, 0);
  int32_t expected_avg = 0;
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    const int vector_size = 50;
    global_vec = leontev_n_average_mpi::getRandomVector(vector_size);
    taskEmplacement(taskDataPar, global_vec, global_avg);
  }
  leontev_n_average_mpi::MPIVecAvgParallel MPIVecAvgParallel(taskDataPar);
  ASSERT_TRUE(MPIVecAvgParallel.ValidationImpl());
  MPIVecAvgParallel.PreProcessingImpl();
  MPIVecAvgParallel.RunImpl();
  MPIVecAvgParallel.PostProcessingImpl();
  expected_avg = std::accumulate(global_vec.begin(), global_vec.end(), 0) / global_vec.size();
  ASSERT_EQ(expected_avg, global_avg[0]);
}
TEST(leontev_n_average_mpi, avg_mpi_0elem) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int32_t> global_avg(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskEmplacement(taskDataPar, global_vec, global_avg);
  }
  leontev_n_average_mpi::MPIVecAvgParallel MPIVecAvgParallel(taskDataPar);
  ASSERT_FALSE(MPIVecAvgParallel.ValidationImpl());
}
TEST(leontev_n_average_mpi, avg_mpi_1000elem) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int32_t> global_avg(1, 0);
  int32_t expected_avg = 0;
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    const int vector_size = 1000;
    global_vec = leontev_n_average_mpi::getRandomVector(vector_size);
    taskEmplacement(taskDataPar, global_vec, global_avg);
  }
  leontev_n_average_mpi::MPIVecAvgParallel MPIVecAvgParallel(taskDataPar);
  ASSERT_TRUE(MPIVecAvgParallel.ValidationImpl());
  MPIVecAvgParallel.PreProcessingImpl();
  MPIVecAvgParallel.RunImpl();
  MPIVecAvgParallel.PostProcessingImpl();
  expected_avg = std::accumulate(global_vec.begin(), global_vec.end(), 0) / global_vec.size();
  ASSERT_EQ(expected_avg, global_avg[0]);
}
TEST(leontev_n_average_mpi, avg_mpi_20000elem) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int32_t> global_avg(1, 0);
  int32_t expected_avg = 0;
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    const int vector_size = 20000;
    global_vec = leontev_n_average_mpi::getRandomVector(vector_size);
    taskEmplacement(taskDataPar, global_vec, global_avg);
  }
  leontev_n_average_mpi::MPIVecAvgParallel MPIVecAvgParallel(taskDataPar);
  ASSERT_TRUE(MPIVecAvgParallel.ValidationImpl());
  MPIVecAvgParallel.PreProcessingImpl();
  MPIVecAvgParallel.RunImpl();
  MPIVecAvgParallel.PostProcessingImpl();
  expected_avg = std::accumulate(global_vec.begin(), global_vec.end(), 0) / global_vec.size();
  ASSERT_EQ(expected_avg, global_avg[0]);
}
TEST(leontev_n_average_mpi, avg_mpi_1elem) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int32_t> global_avg(1, 0);
  int32_t expected_avg = 0;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    const int vector_size = 1;
    global_vec = leontev_n_average_mpi::getRandomVector(vector_size);
    taskEmplacement(taskDataPar, global_vec, global_avg);
  }
  leontev_n_average_mpi::MPIVecAvgParallel MPIVecAvgParallel(taskDataPar);
  ASSERT_TRUE(MPIVecAvgParallel.ValidationImpl());
  MPIVecAvgParallel.PreProcessingImpl();
  MPIVecAvgParallel.RunImpl();
  MPIVecAvgParallel.PostProcessingImpl();
  expected_avg = std::accumulate(global_vec.begin(), global_vec.end(), 0) / global_vec.size();
  ASSERT_EQ(expected_avg, global_avg[0]);
}
