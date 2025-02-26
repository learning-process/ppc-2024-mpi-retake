#include "seq/muradov_k_odd_even_batcher_sort/include/ops_seq.hpp"

#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <random>
#include <vector>

namespace muradov_k_odd_even_batcher_sort {

std::vector<int> RandomVector(int size) {
  std::mt19937 rng(static_cast<unsigned int>(time(nullptr)));
  std::vector<int> v(size);
  for (int i = 0; i < size; ++i) {
    v[i] = static_cast<int>(rng() % 1000);
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
}  // anonymous namespace

void QSort(std::vector<int>& v, int l, int r) { QSortImpl(v, l, r); }

void OddEvenBatcherSort(std::vector<int>& v) {
  if (!v.empty()) {
    QSort(v, 0, static_cast<int>(v.size()) - 1);
  }
}

}  // namespace muradov_k_odd_even_batcher_sort
