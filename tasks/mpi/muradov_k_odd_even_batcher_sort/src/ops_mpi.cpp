#define OMPI_SKIP_MPICXX

#include "mpi/muradov_k_odd_even_batcher_sort/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <random>
#include <utility>
#include <vector>

namespace muradov_k_odd_even_batcher_sort {

std::vector<int> RandomVector(int size) {
  std::mt19937 rng(static_cast<unsigned int>(time(nullptr)));
  std::vector<int> v(size);
  for (int i = 0; i < size; ++i) {
    v[i] = static_cast<int>(rng()) % 1000;
  }
  return v;
}

namespace {

// Partition function for quicksort.
int Partition(std::vector<int>& v, int l, int r) {
  int pivot = v[r];
  int i = l - 1;
  for (int j = l; j < r; ++j) {
    if (v[j] <= pivot) {
      ++i;
      std::swap(v[i], v[j]);
    }
  }
  std::swap(v[i + 1], v[r]);
  return i + 1;
}

void QSortImpl(std::vector<int>& v, int l, int r) {
  if (l < r) {
    int p = Partition(v, l, r);
    QSortImpl(v, l, p - 1);
    QSortImpl(v, p + 1, r);
  }
}

// Extracted merge for the descending case.
void mergeDescending(std::vector<int>& local_part, const std::vector<int>& neighbor_part, std::vector<int>& tmp) {
  int part_size = static_cast<int>(local_part.size());
  int idx1 = part_size - 1;
  int idx2 = part_size - 1;
  for (int j = part_size - 1; j >= 0; --j) {
    if (idx1 >= 0 && idx2 >= 0) {
      if (local_part[idx1] > neighbor_part[idx2]) {
        tmp[j] = local_part[idx1--];
      } else {
        tmp[j] = neighbor_part[idx2--];
      }
    } else if (idx1 >= 0) {
      tmp[j] = local_part[idx1--];
    } else {
      tmp[j] = neighbor_part[idx2--];
    }
  }
  local_part = tmp;
}

// Extracted merge for the ascending case.
void mergeAscending(std::vector<int>& local_part, const std::vector<int>& neighbor_part, std::vector<int>& tmp) {
  int part_size = static_cast<int>(local_part.size());
  int idx1 = 0;
  int idx2 = 0;
  for (int j = 0; j < part_size; ++j) {
    if (idx1 < part_size && idx2 < part_size) {
      if (local_part[idx1] < neighbor_part[idx2]) {
        tmp[j] = local_part[idx1++];
      } else {
        tmp[j] = neighbor_part[idx2++];
      }
    } else if (idx1 < part_size) {
      tmp[j] = local_part[idx1++];
    } else {
      tmp[j] = neighbor_part[idx2++];
    }
  }
  local_part = tmp;
}

void OddEvenMerge(const std::vector<int>& l, const std::vector<int>& r, std::vector<std::pair<int, int>>& schedule) {
  int size = static_cast<int>(l.size() + r.size());
  if (size <= 1) {
    return;
  }
  if (size == 2) {
    schedule.emplace_back(l[0], r[0]);
    return;
  }
  std::vector<int> l_even;
  std::vector<int> l_odd;
  std::vector<int> r_even;
  std::vector<int> r_odd;
  std::vector<int> res;
  for (size_t i = 0; i < l.size(); ++i) {
    if (i % 2 == 0) {
      l_even.push_back(l[i]);
    } else {
      l_odd.push_back(l[i]);
    }
  }
  for (size_t i = 0; i < r.size(); ++i) {
    if (i % 2 == 0) {
      r_even.push_back(r[i]);
    } else {
      r_odd.push_back(r[i]);
    }
  }
  OddEvenMerge(l_odd, r_odd, schedule);
  OddEvenMerge(l_even, r_even, schedule);
  res.insert(res.end(), l.begin(), l.end());
  res.insert(res.end(), r.begin(), r.end());
  for (int i = 1; i < size - 1; i += 2) {
    schedule.emplace_back(res[i], res[i + 1]);
  }
}

void Allocation(const std::vector<int>& v, std::vector<std::pair<int, int>>& schedule) {
  int size = static_cast<int>(v.size());
  if (size <= 1) {
    return;
  }
  std::vector<int> l(v.begin(), v.begin() + size / 2);
  std::vector<int> r(v.begin() + size / 2, v.end());
  Allocation(l, schedule);
  Allocation(r, schedule);
  OddEvenMerge(l, r, schedule);
}

std::vector<std::pair<int, int>> BuildAllocation(int proc_num) {
  std::vector<int> v(proc_num);
  for (int i = 0; i < proc_num; ++i) {
    v[i] = i;
  }
  std::vector<std::pair<int, int>> schedule;
  Allocation(v, schedule);
  return schedule;
}

}  // anonymous namespace

void QSort(std::vector<int>& v, int l, int r) { QSortImpl(v, l, r); }

void OddEvenBatcherSort(std::vector<int>& v) {
  int proc_rank = 0;
  int proc_count = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &proc_count);

  if (proc_count <= 1 || static_cast<int>(v.size()) <= proc_count) {
    if (proc_rank == 0 && !v.empty()) {
      QSort(v, 0, static_cast<int>(v.size()) - 1);
    }
    return;
  }

  std::vector<std::pair<int, int>> schedule = BuildAllocation(proc_count);

  int padding_count = 0;
  if (proc_rank == 0) {
    while (v.size() % proc_count != 0) {
      v.push_back(1337);
      ++padding_count;
    }
  }

  int enlarged_size = 0;
  if (proc_rank == 0) {
    enlarged_size = static_cast<int>(v.size());
  }
  MPI_Bcast(&enlarged_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  int part_size = enlarged_size / proc_count;
  std::vector<int> local_part(part_size);
  MPI_Scatter(v.data(), part_size, MPI_INT, local_part.data(), part_size, MPI_INT, 0, MPI_COMM_WORLD);

  if (!local_part.empty()) {
    QSort(local_part, 0, static_cast<int>(local_part.size()) - 1);
  }

  std::vector<int> tmp(part_size);
  std::vector<int> neighbor_part(part_size);

  for (size_t i = 0; i < schedule.size(); ++i) {
    int proc_first = schedule[i].first;
    int proc_second = schedule[i].second;
    MPI_Status status;
    if (proc_rank == proc_second) {
      MPI_Recv(neighbor_part.data(), part_size, MPI_INT, proc_first, 0, MPI_COMM_WORLD, &status);
      MPI_Send(local_part.data(), part_size, MPI_INT, proc_first, 0, MPI_COMM_WORLD);
      mergeDescending(local_part, neighbor_part, tmp);
    } else if (proc_rank == proc_first) {
      MPI_Send(local_part.data(), part_size, MPI_INT, proc_second, 0, MPI_COMM_WORLD);
      MPI_Recv(neighbor_part.data(), part_size, MPI_INT, proc_second, 0, MPI_COMM_WORLD, &status);
      mergeAscending(local_part, neighbor_part, tmp);
    }
  }

  MPI_Gather(local_part.data(), part_size, MPI_INT, v.data(), part_size, MPI_INT, 0, MPI_COMM_WORLD);

  if (proc_rank == 0 && padding_count > 0) {
    v.resize(enlarged_size - padding_count);
  }
}

}  // namespace muradov_k_odd_even_batcher_sort
