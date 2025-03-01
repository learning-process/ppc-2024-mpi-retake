#include <gtest/gtest.h>

#include <climits>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"
#include "core/util/include/util.hpp"
#include "mpi/sedova_o_min_of_vector_elements/include/ops_mpi.hpp"

TEST(sedova_o_min_of_vector_elements_mpi, test_10x10) {
  const int rows = 10;
  const int columns = 10;
  const int min = -500;
  const int max = 500;

  boost::mpi::communicator world_;
  std::vector<std::vector<int>> global_matrix;
  std::vector<int> output(1, INT_MAX);
  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();
  if(world_.rank() == 0) {
    global_matrix = sedova_o_min_of_vector_elements_mpi::getRandomMatrix(rows, columns, min, max);
    for (unsigned int i = 0; i < global_matrix.size(); i++)
      task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_matrix[i].data()));
    task_data_par->inputs_count.emplace_back(rows);
    task_data_par->inputs_count.emplace_back(columns);

    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
    task_data_par->outputs_count.emplace_back(output.size());
  }

  sedova_o_min_of_vector_elements_mpi::TestTaskMPI test_task_parallel(task_data_par);
  ASSERT_EQ(test_task_parallel.ValidationImpl(), true);
  test_task_parallel.PreProcessingImpl();
  test_task_parallel.RunImpl();
  test_task_parallel.PostProcessingImpl();

  if (world_.rank() == 0) {
    // Create data
    std::vector<int> output1(1, INT_MAX);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
    for (unsigned int i = 0; i < global_matrix.size(); i++)
      task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_matrix[i].data()));
    task_data_seq->inputs_count.emplace_back(rows);
    task_data_seq->inputs_count.emplace_back(columns);

    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(output1.data()));
    task_data_seq->outputs_count.emplace_back(output1.size());

    // Create Task
    sedova_o_min_of_vector_elements_mpi::TestTaskSequential test_task_sequential(task_data_seq);
    ASSERT_EQ(test_task_sequential.ValidationImpl(), true);
    test_task_sequential.PreProcessingImpl();
    test_task_sequential.RunImpl();
    test_task_sequential.PostProcessingImpl();

    ASSERT_EQ(output1[0], output[0]);
  }
}