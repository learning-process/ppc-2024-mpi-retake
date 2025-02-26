#ifndef MURADOV_K_ODD_EVEN_BATCHER_SORT_OPS_MPI_HPP
#define MURADOV_K_ODD_EVEN_BATCHER_SORT_OPS_MPI_HPP
#ifndef OMPI_SKIP_MPICXX
#define OMPI_SKIP_MPICXX
#endif

#include <mpi.h>

#include <vector>

namespace muradov_k_odd_even_batcher_sort {

std::vector<int> random_vector(int size);
void q_sort(std::vector<int>& v, int l, int r);
void odd_even_batcher_sort(std::vector<int>& v);

}  // namespace muradov_k_odd_even_batcher_sort

#endif  // MURADOV_K_ODD_EVEN_BATCHER_SORT_OPS_MPI_HPP
