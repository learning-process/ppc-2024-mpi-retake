#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "mpi/dudchenko_o_sum_values_by_cols/include/ops_mpi.hpp"

TEST(dudchenko_o_sum_values_by_cols_mpi, test_pipeline_run) {
  boost::mpi::communicator world;

  int cols = 12000;
  int rows = 12000;

  std::vector<int> in(cols * rows, 0);
  std::vector<int> expect(cols, 0);
  std::vector<int> out(cols, 0);

  for (int j = 0; j < cols; j++) {
    int tmp_sum = 0;
    for (int i = 0; i < rows; i++) {
      int value = (i * cols) + j;
      in[(i * cols) + j] = value;
      tmp_sum += value;
    }
    expect[j] = tmp_sum;
  }

  auto task_data_par = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_par->inputs_count = {static_cast<unsigned int>(in.size()), static_cast<unsigned int>(rows),
                                   static_cast<unsigned int>(cols)};
    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_par->outputs_count.emplace_back(static_cast<unsigned int>(out.size()));
  }

  auto test_task_par = std::make_shared<dudchenko_o_sum_values_by_cols_mpi::SumValByColsMpi>(task_data_par);
  ASSERT_TRUE(test_task_par->ValidationImpl());
  test_task_par->PreProcessingImpl();
  test_task_par->RunImpl();
  test_task_par->PostProcessingImpl();

  if (world.rank() == 0) {
    ASSERT_EQ(expect, out);
  }
}

TEST(dudchenko_o_sum_values_by_cols_mpi, test_task_run) {
  boost::mpi::communicator world;

  int cols = 12000;
  int rows = 12000;

  std::vector<int> in(cols * rows, 0);
  std::vector<int> expect(cols, 0);
  std::vector<int> out(cols, 0);

  for (int j = 0; j < cols; j++) {
    int tmp_sum = 0;
    for (int i = 0; i < rows; i++) {
      int value = (i * cols) + j;
      in[(i * cols) + j] = value;
      tmp_sum += value;
    }
    expect[j] = tmp_sum;
  }

  auto task_data_par = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_par->inputs_count = {static_cast<unsigned int>(in.size()), static_cast<unsigned int>(rows),
                                   static_cast<unsigned int>(cols)};
    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_par->outputs_count.emplace_back(static_cast<unsigned int>(out.size()));
  }

  auto test_task_par = std::make_shared<dudchenko_o_sum_values_by_cols_mpi::SumValByColsMpi>(task_data_par);
  ASSERT_TRUE(test_task_par->ValidationImpl());
  test_task_par->PreProcessingImpl();
  test_task_par->RunImpl();
  test_task_par->PostProcessingImpl();

  if (world.rank() == 0) {
    ASSERT_EQ(expect, out);
  }
}
