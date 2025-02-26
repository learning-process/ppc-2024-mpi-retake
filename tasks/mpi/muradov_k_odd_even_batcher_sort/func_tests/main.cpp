#define OMPI_SKIP_MPICXX

#include <gtest/gtest.h>
#include <mpi.h>

#include <algorithm>
#include <vector>

#include "mpi/muradov_k_odd_even_batcher_sort/include/ops_mpi.hpp"

namespace mk = muradov_k_odd_even_batcher_sort;

TEST(muradov_k_odd_even_batcher_sort_func, positive_values) {
  int proc_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
  std::vector<int> v = {9, 8, 7, 1, 5};
  std::vector<int> expected = {1, 5, 7, 8, 9};
  mk::OddEvenBatcherSort(v);
  if (proc_rank == 0) {
    ASSERT_EQ(v, expected);
  }
}

TEST(muradov_k_odd_even_batcher_sort_func, negative_values) {
  int proc_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
  std::vector<int> v = {-7, -8, -9, -1, -5};
  std::vector<int> expected = {-9, -8, -7, -5, -1};
  mk::OddEvenBatcherSort(v);
  if (proc_rank == 0) {
    ASSERT_EQ(v, expected);
  }
}

TEST(muradov_k_odd_even_batcher_sort_func, mixed_values) {
  int proc_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
  std::vector<int> v = {-7, 8, 0, -1, 5};
  std::vector<int> expected = {-7, -1, 0, 5, 8};
  mk::OddEvenBatcherSort(v);
  if (proc_rank == 0) {
    ASSERT_EQ(v, expected);
  }
}

TEST(muradov_k_odd_even_batcher_sort_func, compare_with_qsort) {
  int proc_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
  int n = 235;
  std::vector<int> v = mk::RandomVector(n);
  std::vector<int> v_copy = v;
  mk::OddEvenBatcherSort(v);
  if (proc_rank == 0) {
    mk::QSort(v_copy, 0, static_cast<int>(v_copy.size()) - 1);
    ASSERT_EQ(v, v_copy);
  }
}

TEST(muradov_k_odd_even_batcher_sort_func, compare_with_std_sort) {
  int proc_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
  int n = 235;
  std::vector<int> v = mk::RandomVector(n);
  std::vector<int> v_copy = v;
  mk::OddEvenBatcherSort(v);
  if (proc_rank == 0) {
    std::ranges::sort(v_copy);
    ASSERT_EQ(v, v_copy);
  }
}
