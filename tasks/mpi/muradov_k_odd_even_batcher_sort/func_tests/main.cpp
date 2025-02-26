#include <gtest/gtest.h>
#include <mpi.h>

#include <algorithm>
#include <vector>

#include "mpi/muradov_k_odd_even_batcher_sort/include/ops_mpi.hpp"

namespace mk = muradov_k_odd_even_batcher_sort;

TEST(muradov_k_odd_even_batcher_sort_func, positive_values) {
  int procRank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &procRank);
  std::vector<int> v = {9, 8, 7, 1, 5};
  std::vector<int> expected = {1, 5, 7, 8, 9};
  mk::odd_even_batcher_sort(v);
  if (procRank == 0) {
    ASSERT_EQ(v, expected);
  }
}

TEST(muradov_k_odd_even_batcher_sort_func, negative_values) {
  int procRank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &procRank);
  std::vector<int> v = {-7, -8, -9, -1, -5};
  std::vector<int> expected = {-9, -8, -7, -5, -1};
  mk::odd_even_batcher_sort(v);
  if (procRank == 0) {
    ASSERT_EQ(v, expected);
  }
}

TEST(muradov_k_odd_even_batcher_sort_func, mixed_values) {
  int procRank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &procRank);
  std::vector<int> v = {-7, 8, 0, -1, 5};
  std::vector<int> expected = {-7, -1, 0, 5, 8};
  mk::odd_even_batcher_sort(v);
  if (procRank == 0) {
    ASSERT_EQ(v, expected);
  }
}

TEST(muradov_k_odd_even_batcher_sort_func, compare_with_qsort) {
  int procRank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &procRank);
  int n = 235;
  std::vector<int> v = mk::random_vector(n);
  std::vector<int> v_copy = v;
  mk::odd_even_batcher_sort(v);
  if (procRank == 0) {
    mk::q_sort(v_copy, 0, static_cast<int>(v_copy.size()) - 1);
    ASSERT_EQ(v, v_copy);
  }
}

TEST(muradov_k_odd_even_batcher_sort_func, compare_with_std_sort) {
  int procRank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &procRank);
  int n = 235;
  std::vector<int> v = mk::random_vector(n);
  std::vector<int> v_copy = v;
  mk::odd_even_batcher_sort(v);
  if (procRank == 0) {
    std::sort(v_copy.begin(), v_copy.end());
    ASSERT_EQ(v, v_copy);
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  MPI_Init(&argc, &argv);
  int result = RUN_ALL_TESTS();
  MPI_Finalize();
  return result;
}
