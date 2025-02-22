// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "mpi/shishkarev_a_sum_of_vector_elements/include/ops_mpi.hpp"

TEST(shishkarev_a_sum_of_vector_elements_mpi, test_pipeline_run) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int32_t> global_sum(1, 0);
  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();
  int count_size_vector = 0;

  if (world.rank() == 0) {
    count_size_vector = 100000000;
    global_vec = std::vector<int>(count_size_vector, 1);
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    task_data_par->inputs_count.emplace_back(global_vec.size());
    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_sum.data()));
    task_data_par->outputs_count.emplace_back(global_sum.size());
  }

  auto mpi_vector_sum_parallel =
      std::make_shared<shishkarev_a_sum_of_vector_elements_mpi::MPIVectorSumParallel>(task_data_par);
  ASSERT_TRUE(mpi_vector_sum_parallel->ValidationImpl());
  mpi_vector_sum_parallel->PreProcessingImpl();
  mpi_vector_sum_parallel->RunImpl();
  mpi_vector_sum_parallel->PostProcessingImpl();

  if (world.rank() == 0) {
    ASSERT_EQ(global_sum[0], count_size_vector);
  }
}

TEST(shishkarev_a_sum_of_vector_elements_mpi, test_task_run) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int32_t> global_sum(1, 0);
  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();
  int count_size_vector = 0;

  if (world.rank() == 0) {
    count_size_vector = 100000000;
    global_vec = std::vector<int>(count_size_vector, 1);
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    task_data_par->inputs_count.emplace_back(global_vec.size());
    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_sum.data()));
    task_data_par->outputs_count.emplace_back(global_sum.size());
  }

  auto mpi_vector_sum_parallel =
      std::make_shared<shishkarev_a_sum_of_vector_elements_mpi::MPIVectorSumParallel>(task_data_par);
  ASSERT_TRUE(mpi_vector_sum_parallel->ValidationImpl());
  mpi_vector_sum_parallel->PreProcessingImpl();
  mpi_vector_sum_parallel->RunImpl();
  mpi_vector_sum_parallel->PostProcessingImpl();

  if (world.rank() == 0) {
    ASSERT_EQ(global_sum[0], count_size_vector);
  }
}