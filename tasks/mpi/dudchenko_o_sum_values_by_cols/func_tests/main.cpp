#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "mpi/dudchenko_o_sum_values_by_cols/include/ops_mpi.hpp"

namespace {
void FillRandom(std::vector<int> &data, int min_val = 0, int max_val = 100) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(min_val, max_val);
  for (auto &val : data) {
    val = dis(gen);
  }
}
}  // namespace

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

TEST(dudchenko_o_sum_values_by_cols_mpi, test_fixed_3x3_matrix) {
  boost::mpi::communicator world;

  int cols = 3;
  int rows = 3;
  std::vector<int> in = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<int> out_par(cols, 0);
  std::vector<int> expect = {12, 15, 18};

  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_par->inputs_count = {static_cast<unsigned int>(in.size()), static_cast<unsigned int>(rows),
                                   static_cast<unsigned int>(cols)};
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

TEST(dudchenko_o_sum_values_by_cols_mpi, test_random_4x5_matrix) {
  boost::mpi::communicator world;

  int cols = 5;
  int rows = 4;
  std::vector<int> in(cols * rows);
  FillRandom(in);
  std::vector<int> out_par(cols, 0);
  std::vector<int> expect(cols, 0);

  if (world.rank() == 0) {
    for (int j = 0; j < cols; ++j) {
      for (int i = 0; i < rows; ++i) {
        expect[j] += in[(i * cols) + j];
      }
    }
  }

  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_par->inputs_count = {static_cast<unsigned int>(in.size()), static_cast<unsigned int>(rows),
                                   static_cast<unsigned int>(cols)};
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

TEST(dudchenko_o_sum_values_by_cols_mpi, test_random_6x7_matrix) {
  boost::mpi::communicator world;

  int cols = 7;
  int rows = 6;
  std::vector<int> in(cols * rows);
  FillRandom(in);
  std::vector<int> out_par(cols, 0);
  std::vector<int> expect(cols, 0);

  if (world.rank() == 0) {
    for (int j = 0; j < cols; ++j) {
      for (int i = 0; i < rows; ++i) {
        expect[j] += in[(i * cols) + j];
      }
    }
  }

  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_par->inputs_count = {static_cast<unsigned int>(in.size()), static_cast<unsigned int>(rows),
                                   static_cast<unsigned int>(cols)};
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

TEST(dudchenko_o_sum_values_by_cols_mpi, test_random_10x10_matrix) {
  boost::mpi::communicator world;

  int cols = 10;
  int rows = 10;
  std::vector<int> in(cols * rows);
  FillRandom(in);
  std::vector<int> out_par(cols, 0);
  std::vector<int> expect(cols, 0);

  if (world.rank() == 0) {
    for (int j = 0; j < cols; ++j) {
      for (int i = 0; i < rows; ++i) {
        expect[j] += in[(i * cols) + j];
      }
    }
  }

  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_par->inputs_count = {static_cast<unsigned int>(in.size()), static_cast<unsigned int>(rows),
                                   static_cast<unsigned int>(cols)};
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

TEST(dudchenko_o_sum_values_by_cols_mpi, test_random_20x5_matrix) {
  boost::mpi::communicator world;

  int cols = 5;
  int rows = 20;
  std::vector<int> in(cols * rows);
  FillRandom(in);
  std::vector<int> out_par(cols, 0);
  std::vector<int> expect(cols, 0);

  if (world.rank() == 0) {
    for (int j = 0; j < cols; ++j) {
      for (int i = 0; i < rows; ++i) {
        expect[j] += in[(i * cols) + j];
      }
    }
  }

  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_par->inputs_count = {static_cast<unsigned int>(in.size()), static_cast<unsigned int>(rows),
                                   static_cast<unsigned int>(cols)};
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

TEST(dudchenko_o_sum_values_by_cols_mpi, test_large_50x50_matrix) {
  boost::mpi::communicator world;

  int cols = 50;
  int rows = 50;
  std::vector<int> in(cols * rows);
  FillRandom(in, 1000, 5000);
  std::vector<int> out_par(cols, 0);
  std::vector<int> expect(cols, 0);

  if (world.rank() == 0) {
    for (int j = 0; j < cols; ++j) {
      for (int i = 0; i < rows; ++i) {
        expect[j] += in[(i * cols) + j];
      }
    }
  }

  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_par->inputs_count = {static_cast<unsigned int>(in.size()), static_cast<unsigned int>(rows),
                                   static_cast<unsigned int>(cols)};
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
