#include "mpi/muradov_k_odd_even_batcher_sort/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <random>
#include <utility>
#include <vector>

namespace muradov_k_odd_even_batcher_sort {

std::vector<int> random_vector(int size) {
  std::mt19937 rng(static_cast<unsigned int>(time(0)));
  std::vector<int> v(size);
  for (int i = 0; i < size; ++i) {
    v[i] = rng() % 1000;
  }
  return v;
}

namespace {
// Partition function for quicksort.
int partition(std::vector<int>& v, int l, int r) {
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

void q_sort_impl(std::vector<int>& v, int l, int r) {
  if (l < r) {
    int p = partition(v, l, r);
    q_sort_impl(v, l, p - 1);
    q_sort_impl(v, p + 1, r);
  }
}
}  // anonymous namespace

void q_sort(std::vector<int>& v, int l, int r) { q_sort_impl(v, l, r); }

// --- Helpers to build the odd–even merge schedule ---
namespace {
void odd_even_merge(const std::vector<int>& l, const std::vector<int>& r, std::vector<std::pair<int, int>>& schedule) {
  int size = static_cast<int>(l.size() + r.size());
  if (size <= 1) return;
  if (size == 2) {
    schedule.push_back({l[0], r[0]});
    return;
  }
  std::vector<int> l_even, l_odd, r_even, r_odd, res;
  for (size_t i = 0; i < l.size(); ++i) {
    if (i % 2 == 0)
      l_even.push_back(l[i]);
    else
      l_odd.push_back(l[i]);
  }
  for (size_t i = 0; i < r.size(); ++i) {
    if (i % 2 == 0)
      r_even.push_back(r[i]);
    else
      r_odd.push_back(r[i]);
  }
  odd_even_merge(l_odd, r_odd, schedule);
  odd_even_merge(l_even, r_even, schedule);
  res.insert(res.end(), l.begin(), l.end());
  res.insert(res.end(), r.begin(), r.end());
  for (int i = 1; i < size - 1; i += 2) {
    schedule.push_back({res[i], res[i + 1]});
  }
}

void allocation(const std::vector<int>& v, std::vector<std::pair<int, int>>& schedule) {
  int size = static_cast<int>(v.size());
  if (size <= 1) return;
  std::vector<int> l(v.begin(), v.begin() + size / 2);
  std::vector<int> r(v.begin() + size / 2, v.end());
  allocation(l, schedule);
  allocation(r, schedule);
  odd_even_merge(l, r, schedule);
}

std::vector<std::pair<int, int>> build_allocation(int procNum) {
  std::vector<int> v(procNum);
  for (int i = 0; i < procNum; ++i) {
    v[i] = i;
  }
  std::vector<std::pair<int, int>> schedule;
  allocation(v, schedule);
  return schedule;
}
}  // anonymous namespace

// --- Parallel odd–even batcher sort ---
void odd_even_batcher_sort(std::vector<int>& v) {
  int procRank = 0, procCount = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &procRank);
  MPI_Comm_size(MPI_COMM_WORLD, &procCount);

  // If only one process or the vector is too small, perform sequential sort.
  if (procCount <= 1 || static_cast<int>(v.size()) <= procCount) {
    if (procRank == 0 && !v.empty()) {
      q_sort(v, 0, static_cast<int>(v.size()) - 1);
    }
    return;
  }

  // Build the communication schedule.
  std::vector<std::pair<int, int>> schedule = build_allocation(procCount);

  int paddingCount = 0;
  if (procRank == 0) {
    // Pad the vector so its size is divisible by procCount.
    while (v.size() % procCount != 0) {
      v.push_back(1337);  // dummy value
      ++paddingCount;
    }
  }

  int enlarged_size = 0;
  if (procRank == 0) {
    enlarged_size = static_cast<int>(v.size());
  }
  MPI_Bcast(&enlarged_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  int partSize = enlarged_size / procCount;
  std::vector<int> localPart(partSize);
  MPI_Scatter(v.data(), partSize, MPI_INT, localPart.data(), partSize, MPI_INT, 0, MPI_COMM_WORLD);

  // Local sort.
  if (!localPart.empty()) {
    q_sort(localPart, 0, static_cast<int>(localPart.size()) - 1);
  }

  // Temporary buffers for merging.
  std::vector<int> tmp(partSize);
  std::vector<int> neighborPart(partSize);

  // Merge according to the schedule.
  for (size_t i = 0; i < schedule.size(); ++i) {
    int procFirst = schedule[i].first;
    int procSecond = schedule[i].second;
    MPI_Status status;
    if (procRank == procSecond) {
      MPI_Recv(neighborPart.data(), partSize, MPI_INT, procFirst, 0, MPI_COMM_WORLD, &status);
      MPI_Send(localPart.data(), partSize, MPI_INT, procFirst, 0, MPI_COMM_WORLD);
      // Merge in descending order.
      int idx1 = partSize - 1, idx2 = partSize - 1;
      for (int j = partSize - 1; j >= 0; --j) {
        if (idx1 >= 0 && idx2 >= 0) {
          if (localPart[idx1] > neighborPart[idx2]) {
            tmp[j] = localPart[idx1--];
          } else {
            tmp[j] = neighborPart[idx2--];
          }
        } else if (idx1 >= 0) {
          tmp[j] = localPart[idx1--];
        } else {
          tmp[j] = neighborPart[idx2--];
        }
      }
      localPart = tmp;
    } else if (procRank == procFirst) {
      MPI_Send(localPart.data(), partSize, MPI_INT, procSecond, 0, MPI_COMM_WORLD);
      MPI_Recv(neighborPart.data(), partSize, MPI_INT, procSecond, 0, MPI_COMM_WORLD, &status);
      // Merge in ascending order.
      int idx1 = 0, idx2 = 0;
      for (int j = 0; j < partSize; ++j) {
        if (idx1 < partSize && idx2 < partSize) {
          if (localPart[idx1] < neighborPart[idx2]) {
            tmp[j] = localPart[idx1++];
          } else {
            tmp[j] = neighborPart[idx2++];
          }
        } else if (idx1 < partSize) {
          tmp[j] = localPart[idx1++];
        } else {
          tmp[j] = neighborPart[idx2++];
        }
      }
      localPart = tmp;
    }
    // Processes not in the current pair simply continue.
  }

  MPI_Gather(localPart.data(), partSize, MPI_INT, v.data(), partSize, MPI_INT, 0, MPI_COMM_WORLD);

  if (procRank == 0 && paddingCount > 0) {
    v.resize(enlarged_size - paddingCount);
  }
}

}  // namespace muradov_k_odd_even_batcher_sort
