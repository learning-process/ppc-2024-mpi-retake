#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "mpi/dudchenko_o_sum_values_by_cols/include/ops_mpi.hpp"

TEST(dudchenko_o_sum_values_by_cols_mpi, test_empty_matrix) {
  boost::mpi::communicator world;

  int cols = 0;
  int rows = 0;

  std::vector<int> in = {};
  std::vector<int> out_par = {};

  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_par->inputs_count.emplace_back(in.size());
    task_data_par->inputs_count.emplace_back(rows);
    task_data_par->inputs_count.emplace_back(cols);
    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_par.data()));
    task_data_par->outputs_count.emplace_back(out_par.size());
  }

  dudchenko_o_sum_values_by_cols_mpi::SumValByColsMpi sum_val_by_cols_mpi(task_data_par);
  ASSERT_EQ(sum_val_by_cols_mpi.ValidationImpl(), true);
  sum_val_by_cols_mpi.PreProcessingImpl();
  sum_val_by_cols_mpi.RunImpl();
  sum_val_by_cols_mpi.PostProcessingImpl();
}

TEST(dudchenko_o_sum_values_by_cols_mpi, test_1x1_matrix) {
  boost::mpi::communicator world;

  int cols = 1;
  int rows = 1;

  std::vector<int> in(cols * rows, 1);
  std::vector<int> out_par(cols, 0);

  std::vector<int> expect(cols, 1);

  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_par->inputs_count.emplace_back(in.size());
    task_data_par->inputs_count.emplace_back(rows);
    task_data_par->inputs_count.emplace_back(cols);
    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_par.data()));
    task_data_par->outputs_count.emplace_back(out_par.size());
  }

  dudchenko_o_sum_values_by_cols_mpi::SumValByColsMpi sum_val_by_cols_mpi(task_data_par);
  ASSERT_EQ(sum_val_by_cols_mpi.ValidationImpl(), true);
  sum_val_by_cols_mpi.PreProcessingImpl();
  sum_val_by_cols_mpi.RunImpl();
  sum_val_by_cols_mpi.PostProcessingImpl();

  if (world.rank() == 0) {
    ASSERT_EQ(out_par, expect);
  }
}

TEST(dudchenko_o_sum_values_by_cols_mpi, test_5x1_matrix) {
  boost::mpi::communicator world;

  int cols = 5;
  int rows = 1;

  std::vector<int> in(cols * rows, 1);
  std::vector<int> out_par(cols, 0);

  std::vector<int> expect(cols, 1);

  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_par->inputs_count.emplace_back(in.size());
    task_data_par->inputs_count.emplace_back(rows);
    task_data_par->inputs_count.emplace_back(cols);
    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_par.data()));
    task_data_par->outputs_count.emplace_back(out_par.size());
  }

  dudchenko_o_sum_values_by_cols_mpi::SumValByColsMpi sum_val_by_cols_mpi(task_data_par);
  ASSERT_EQ(sum_val_by_cols_mpi.ValidationImpl(), true);
  sum_val_by_cols_mpi.PreProcessingImpl();
  sum_val_by_cols_mpi.RunImpl();
  sum_val_by_cols_mpi.PostProcessingImpl();

  if (world.rank() == 0) {
    ASSERT_EQ(out_par, expect);
  }
}
