#ifndef MURADOV_K_ODD_EVEN_BATCHER_SORT_OPS_SEQ_HPP
#define MURADOV_K_ODD_EVEN_BATCHER_SORT_OPS_SEQ_HPP

#include <vector>

namespace muradov_k_odd_even_batcher_sort {

std::vector<int> RandomVector(int size);
void QSort(std::vector<int>& v, int l, int r);
void OddEvenBatcherSort(std::vector<int>& v);

}  // namespace muradov_k_odd_even_batcher_sort

#endif  // MURADOV_K_ODD_EVEN_BATCHER_SORT_OPS_SEQ_HPP
