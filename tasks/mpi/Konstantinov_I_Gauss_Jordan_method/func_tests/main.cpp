#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "mpi/Konstantinov_I_Gauss_Jordan_method/include/ops_mpi.hpp"

namespace konstantinov_i_gauss_jordan_method_mpi {

std::vector<double> GetRandomMatrix(int rows, int cols, double min = -20.0, double max = 20.0) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_real_distribution<double> dist(min, max);
  std::vector<double> matrix(rows * cols);
  for (int i = 0; i < rows * cols; i++) {
    matrix[i] = dist(gen);
  }
  return matrix;
}

}  // namespace konstantinov_i_gauss_jordan_method_mpi

TEST(konstantinov_i_gauss_jordan_method_mpi, simple_three_not_solve_at_1_iter) {
  boost::mpi::communicator world;

  std::vector<double> global_matrix;
  int n;
  std::vector<double> global_result;

  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_matrix = {0,  2,  3,  4, 5,  6,  0,  8,  9,  10, 11, 12, 0,  14, 15,
                     16, 17, 18, 0, 20, 21, 22, 23, 24, 0,  26, 27, 28, 29, 30};
    n = 5;

    global_result.resize(n * (n + 1));

    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    task_data_par->inputs_count.emplace_back(global_matrix.size());

    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
    task_data_par->inputs_count.emplace_back(1);

    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
    task_data_par->outputs_count.emplace_back(global_result.size());
  }

  auto task_parallel = std::make_shared<konstantinov_i_gauss_jordan_method_mpi::GaussJordanMethodMPI>(task_data_par);
  if (world.rank() == 0) {
    EXPECT_FALSE(task_parallel->ValidationImpl());
  } else {
    EXPECT_TRUE(true);
  }
}

TEST(konstantinov_i_gauss_jordan_method_mpi, simple_five_not_solve_at_2_iter) {
  boost::mpi::communicator world;

  std::vector<double> global_matrix;
  int n;
  std::vector<double> global_result;

  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_matrix = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
                     16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30};
    n = 5;

    global_result.resize(n * (n + 1));

    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    task_data_par->inputs_count.emplace_back(global_matrix.size());

    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
    task_data_par->inputs_count.emplace_back(1);

    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
    task_data_par->outputs_count.emplace_back(global_result.size());
  }

  auto task_parallel = std::make_shared<konstantinov_i_gauss_jordan_method_mpi::GaussJordanMethodMPI>(task_data_par);
  if (world.rank() == 0) {
    EXPECT_FALSE(task_parallel->ValidationImpl());
  } else {
    EXPECT_TRUE(true);
  }
}

TEST(konstantinov_i_gauss_jordan_method_mpi, random_three) {
  boost::mpi::communicator world;

  std::vector<double> global_matrix;
  int n;
  std::vector<double> global_result;

  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    n = 3;
    global_matrix = konstantinov_i_gauss_jordan_method_mpi::GetRandomMatrix(n, n + 1);

    global_result.resize(n * (n + 1));

    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    task_data_par->inputs_count.emplace_back(global_matrix.size());

    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
    task_data_par->inputs_count.emplace_back(1);

    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
    task_data_par->outputs_count.emplace_back(global_result.size());
  }

  auto task_parallel = std::make_shared<konstantinov_i_gauss_jordan_method_mpi::GaussJordanMethodMPI>(task_data_par);
  bool isNonSingular = task_parallel->ValidationImpl();
  if (isNonSingular) {
    task_parallel->PreProcessingImpl();
    bool parRunRes = task_parallel->RunImpl();
    task_parallel->PostProcessingImpl();

    if (world.rank() == 0) {
      std::vector<double> seq_result(global_result.size(), 0);

      auto task_data_seq = std::make_shared<ppc::core::TaskData>();
      task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
      task_data_seq->inputs_count.emplace_back(global_matrix.size());

      task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
      task_data_seq->inputs_count.emplace_back(1);

      task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(seq_result.data()));
      task_data_seq->outputs_count.emplace_back(seq_result.size());

      auto task_sequential =
          std::make_shared<konstantinov_i_gauss_jordan_method_mpi::GaussJordanMethodSeq>(task_data_seq);
      ASSERT_TRUE(task_sequential->ValidationImpl());
      task_sequential->PreProcessingImpl();
      bool seqRunRes = task_sequential->RunImpl();
      task_sequential->PostProcessingImpl();

      if (seqRunRes && parRunRes) {
        ASSERT_EQ(global_result.size(), seq_result.size());
        EXPECT_EQ(global_result, seq_result);
      } else {
        EXPECT_EQ(seqRunRes, parRunRes);
      }
    }
  } else {
    EXPECT_TRUE(true);
  }
}

TEST(konstantinov_i_gauss_jordan_method_mpi, random_four) {
  boost::mpi::communicator world;

  std::vector<double> global_matrix;
  int n;
  std::vector<double> global_result;

  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    n = 4;
    global_matrix = konstantinov_i_gauss_jordan_method_mpi::GetRandomMatrix(n, n + 1, -30.0);

    global_result.resize(n * (n + 1));

    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    task_data_par->inputs_count.emplace_back(global_matrix.size());

    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
    task_data_par->inputs_count.emplace_back(1);

    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
    task_data_par->outputs_count.emplace_back(global_result.size());
  }

  auto task_parallel = std::make_shared<konstantinov_i_gauss_jordan_method_mpi::GaussJordanMethodMPI>(task_data_par);
  bool isNonSingular = task_parallel->ValidationImpl();
  if (isNonSingular) {
    task_parallel->PreProcessingImpl();
    bool parRunRes = task_parallel->RunImpl();
    task_parallel->PostProcessingImpl();

    if (world.rank() == 0) {
      std::vector<double> seq_result(global_result.size(), 0);

      auto task_data_seq = std::make_shared<ppc::core::TaskData>();
      task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
      task_data_seq->inputs_count.emplace_back(global_matrix.size());

      task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
      task_data_seq->inputs_count.emplace_back(1);

      task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(seq_result.data()));
      task_data_seq->outputs_count.emplace_back(seq_result.size());

      auto task_sequential =
          std::make_shared<konstantinov_i_gauss_jordan_method_mpi::GaussJordanMethodSeq>(task_data_seq);
      ASSERT_TRUE(task_sequential->ValidationImpl());
      task_sequential->PreProcessingImpl();
      bool seqRunRes = task_sequential->RunImpl();
      task_sequential->PostProcessingImpl();

      if (seqRunRes && parRunRes) {
        ASSERT_EQ(global_result.size(), seq_result.size());
        EXPECT_EQ(global_result, seq_result);
      } else {
        EXPECT_EQ(seqRunRes, parRunRes);
      }
    }
  } else {
    EXPECT_TRUE(true);
  }
}

TEST(konstantinov_i_gauss_jordan_method_mpi, random_five) {
  boost::mpi::communicator world;

  std::vector<double> global_matrix;
  int n;
  std::vector<double> global_result;

  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    n = 5;
    global_matrix = konstantinov_i_gauss_jordan_method_mpi::GetRandomMatrix(n, n + 1, -100.0, 200.0);

    global_result.resize(n * (n + 1));

    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    task_data_par->inputs_count.emplace_back(global_matrix.size());

    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
    task_data_par->inputs_count.emplace_back(1);

    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
    task_data_par->outputs_count.emplace_back(global_result.size());
  }

  auto task_parallel = std::make_shared<konstantinov_i_gauss_jordan_method_mpi::GaussJordanMethodMPI>(task_data_par);
  bool isNonSingular = task_parallel->ValidationImpl();
  if (isNonSingular) {
    task_parallel->PreProcessingImpl();
    bool parRunRes = task_parallel->RunImpl();
    task_parallel->PostProcessingImpl();

    if (world.rank() == 0) {
      std::vector<double> seq_result(global_result.size(), 0);

      auto task_data_seq = std::make_shared<ppc::core::TaskData>();
      task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
      task_data_seq->inputs_count.emplace_back(global_matrix.size());

      task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
      task_data_seq->inputs_count.emplace_back(1);

      task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(seq_result.data()));
      task_data_seq->outputs_count.emplace_back(seq_result.size());

      auto task_sequential =
          std::make_shared<konstantinov_i_gauss_jordan_method_mpi::GaussJordanMethodSeq>(task_data_seq);
      ASSERT_TRUE(task_sequential->ValidationImpl());
      task_sequential->PreProcessingImpl();
      bool seqRunRes = task_sequential->RunImpl();
      task_sequential->PostProcessingImpl();

      if (seqRunRes && parRunRes) {
        ASSERT_EQ(global_result.size(), seq_result.size());
        EXPECT_EQ(global_result, seq_result);
      } else {
        EXPECT_EQ(seqRunRes, parRunRes);
      }
    }
  } else {
    EXPECT_TRUE(true);
  }
}

TEST(konstantinov_i_gauss_jordan_method_mpi, random_six) {
  boost::mpi::communicator world;

  std::vector<double> global_matrix;
  int n;
  std::vector<double> global_result;

  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    n = 6;
    global_matrix = konstantinov_i_gauss_jordan_method_mpi::GetRandomMatrix(n, n + 1, -10.0, 10.0);

    global_result.resize(n * (n + 1));

    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    task_data_par->inputs_count.emplace_back(global_matrix.size());

    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
    task_data_par->inputs_count.emplace_back(1);

    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
    task_data_par->outputs_count.emplace_back(global_result.size());
  }

  auto task_parallel = std::make_shared<konstantinov_i_gauss_jordan_method_mpi::GaussJordanMethodMPI>(task_data_par);
  bool isNonSingular = task_parallel->ValidationImpl();
  if (isNonSingular) {
    task_parallel->PreProcessingImpl();
    bool parRunRes = task_parallel->RunImpl();
    task_parallel->PostProcessingImpl();

    if (world.rank() == 0) {
      std::vector<double> seq_result(global_result.size(), 0);

      auto task_data_seq = std::make_shared<ppc::core::TaskData>();
      task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
      task_data_seq->inputs_count.emplace_back(global_matrix.size());

      task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
      task_data_seq->inputs_count.emplace_back(1);

      task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(seq_result.data()));
      task_data_seq->outputs_count.emplace_back(seq_result.size());

      auto task_sequential =
          std::make_shared<konstantinov_i_gauss_jordan_method_mpi::GaussJordanMethodSeq>(task_data_seq);
      ASSERT_TRUE(task_sequential->ValidationImpl());
      task_sequential->PreProcessingImpl();
      bool seqRunRes = task_sequential->RunImpl();
      task_sequential->PostProcessingImpl();

      if (seqRunRes && parRunRes) {
        ASSERT_EQ(global_result.size(), seq_result.size());
        EXPECT_EQ(global_result, seq_result);
      } else {
        EXPECT_EQ(seqRunRes, parRunRes);
      }
    }
  } else {
    EXPECT_TRUE(true);
  }
}

TEST(konstantinov_i_gauss_jordan_method_mpi, random_seven) {
  boost::mpi::communicator world;

  std::vector<double> global_matrix;
  int n;
  std::vector<double> global_result;

  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    n = 7;
    global_matrix = konstantinov_i_gauss_jordan_method_mpi::GetRandomMatrix(n, n + 1, -7.0, 1000.0);

    global_result.resize(n * (n + 1));

    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    task_data_par->inputs_count.emplace_back(global_matrix.size());

    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
    task_data_par->inputs_count.emplace_back(1);

    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
    task_data_par->outputs_count.emplace_back(global_result.size());
  }

  auto task_parallel = std::make_shared<konstantinov_i_gauss_jordan_method_mpi::GaussJordanMethodMPI>(task_data_par);
  bool isNonSingular = task_parallel->ValidationImpl();
  if (isNonSingular) {
    task_parallel->PreProcessingImpl();
    bool parRunRes = task_parallel->RunImpl();
    task_parallel->PostProcessingImpl();

    if (world.rank() == 0) {
      std::vector<double> seq_result(global_result.size(), 0);

      auto task_data_seq = std::make_shared<ppc::core::TaskData>();
      task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
      task_data_seq->inputs_count.emplace_back(global_matrix.size());

      task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
      task_data_seq->inputs_count.emplace_back(1);

      task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(seq_result.data()));
      task_data_seq->outputs_count.emplace_back(seq_result.size());

      auto task_sequential =
          std::make_shared<konstantinov_i_gauss_jordan_method_mpi::GaussJordanMethodSeq>(task_data_seq);
      ASSERT_TRUE(task_sequential->ValidationImpl());
      task_sequential->PreProcessingImpl();
      bool seqRunRes = task_sequential->RunImpl();
      task_sequential->PostProcessingImpl();

      if (seqRunRes && parRunRes) {
        ASSERT_EQ(global_result.size(), seq_result.size());
        EXPECT_EQ(global_result, seq_result);
      } else {
        EXPECT_EQ(seqRunRes, parRunRes);
      }
    }
  } else {
    EXPECT_TRUE(true);
  }
}

TEST(konstantinov_i_gauss_jordan_method_mpi, random_ten) {
  boost::mpi::communicator world;

  std::vector<double> global_matrix;
  int n;
  std::vector<double> global_result;

  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    n = 10;
    global_matrix = konstantinov_i_gauss_jordan_method_mpi::GetRandomMatrix(n, n + 1);

    global_result.resize(n * (n + 1));

    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    task_data_par->inputs_count.emplace_back(global_matrix.size());

    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
    task_data_par->inputs_count.emplace_back(1);

    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
    task_data_par->outputs_count.emplace_back(global_result.size());
  }

  auto task_parallel = std::make_shared<konstantinov_i_gauss_jordan_method_mpi::GaussJordanMethodMPI>(task_data_par);
  bool isNonSingular = task_parallel->ValidationImpl();
  if (isNonSingular) {
    task_parallel->PreProcessingImpl();
    bool parRunRes = task_parallel->RunImpl();
    task_parallel->PostProcessingImpl();

    if (world.rank() == 0) {
      std::vector<double> seq_result(global_result.size(), 0);

      auto task_data_seq = std::make_shared<ppc::core::TaskData>();
      task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
      task_data_seq->inputs_count.emplace_back(global_matrix.size());

      task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
      task_data_seq->inputs_count.emplace_back(1);

      task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(seq_result.data()));
      task_data_seq->outputs_count.emplace_back(seq_result.size());

      auto task_sequential =
          std::make_shared<konstantinov_i_gauss_jordan_method_mpi::GaussJordanMethodSeq>(task_data_seq);
      ASSERT_TRUE(task_sequential->ValidationImpl());
      task_sequential->PreProcessingImpl();
      bool seqRunRes = task_sequential->RunImpl();
      task_sequential->PostProcessingImpl();

      if (seqRunRes && parRunRes) {
        ASSERT_EQ(global_result.size(), seq_result.size());
        EXPECT_EQ(global_result, seq_result);
      } else {
        EXPECT_EQ(seqRunRes, parRunRes);
      }
    }
  } else {
    EXPECT_TRUE(true);
  }
}

TEST(konstantinov_i_gauss_jordan_method_mpi, random_eleven) {
  boost::mpi::communicator world;

  std::vector<double> global_matrix;
  int n;
  std::vector<double> global_result;

  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    n = 11;
    global_matrix = konstantinov_i_gauss_jordan_method_mpi::GetRandomMatrix(n, n + 1);

    global_result.resize(n * (n + 1));

    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    task_data_par->inputs_count.emplace_back(global_matrix.size());

    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
    task_data_par->inputs_count.emplace_back(1);

    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
    task_data_par->outputs_count.emplace_back(global_result.size());
  }

  auto task_parallel = std::make_shared<konstantinov_i_gauss_jordan_method_mpi::GaussJordanMethodMPI>(task_data_par);
  bool isNonSingular = task_parallel->ValidationImpl();
  if (isNonSingular) {
    task_parallel->PreProcessingImpl();
    bool parRunRes = task_parallel->RunImpl();
    task_parallel->PostProcessingImpl();

    if (world.rank() == 0) {
      std::vector<double> seq_result(global_result.size(), 0);

      auto task_data_seq = std::make_shared<ppc::core::TaskData>();
      task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
      task_data_seq->inputs_count.emplace_back(global_matrix.size());

      task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
      task_data_seq->inputs_count.emplace_back(1);

      task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(seq_result.data()));
      task_data_seq->outputs_count.emplace_back(seq_result.size());

      auto task_sequential =
          std::make_shared<konstantinov_i_gauss_jordan_method_mpi::GaussJordanMethodSeq>(task_data_seq);
      ASSERT_TRUE(task_sequential->ValidationImpl());
      task_sequential->PreProcessingImpl();
      bool seqRunRes = task_sequential->RunImpl();
      task_sequential->PostProcessingImpl();

      if (seqRunRes && parRunRes) {
        ASSERT_EQ(global_result.size(), seq_result.size());
        EXPECT_EQ(global_result, seq_result);
      } else {
        EXPECT_EQ(seqRunRes, parRunRes);
      }
    }
  } else {
    EXPECT_TRUE(true);
  }
}