#include <algorithm>
#include <boost/mpi/collectives/broadcast.hpp>
#include <boost/mpi/collectives/gatherv.hpp>
#include <boost/mpi/collectives/scatter.hpp>
#include <boost/mpi/collectives/scatterv.hpp>
#include <cstring>
#include <utility>
#include <vector>

#include "mpi/somov_i_ribbon_hor_scheme_only_mat_a/include/somov_i_ribbon_hor_scheme_only_mat_a_mpi.hpp"
namespace somov_i_ribbon_hor_scheme_only_mat_a_mpi {
void LiterallyMult(const std::vector<int>& a, const std::vector<int>& b, std::vector<int>& c, int a_c, int a_r,
                   int b_c) {
  for (int i = 0; i < a_r; ++i) {
    for (int j = 0; j < b_c; ++j) {
      for (int p = 0; p < a_c; ++p) {
        c[(i * b_c) + j] += a[(i * a_c) + p] * b[(p * b_c) + j];
      }
    }
  }
}
bool RibbonHorSchemeOnlyMatA::PreProcessingImpl() {
  if (world_.rank() == 0) {
    a_c_ = static_cast<int>(task_data->inputs_count[0]);
    a_r_ = static_cast<int>(task_data->inputs_count[1]);
    b_c_ = static_cast<int>(task_data->inputs_count[2]);
    b_r_ = static_cast<int>(task_data->inputs_count[3]);

    a_.resize(a_c_ * a_r_);
    b_.resize(b_c_ * b_r_);
    c_.resize(a_r_ * b_c_);

    std::ranges::copy(reinterpret_cast<int*>(task_data->inputs[0]),
                      reinterpret_cast<int*>(task_data->inputs[0]) + (a_c_ * a_r_), a_.begin());

    std::ranges::copy(reinterpret_cast<int*>(task_data->inputs[1]),
                      reinterpret_cast<int*>(task_data->inputs[1]) + (b_c_ * b_r_), b_.begin());
  }

  return true;
}

bool RibbonHorSchemeOnlyMatA::ValidationImpl() {
  if (world_.rank() == 0) {
    return (task_data->outputs_count[0] > 0 &&
            static_cast<int>(task_data->inputs_count[0]) == static_cast<int>(task_data->inputs_count[3]));
  }
  return true;
}

bool RibbonHorSchemeOnlyMatA::RunImpl() {
  int id = world_.rank();
  int size = world_.size();
  std::vector<int> local_data;
  std::vector<int> send_counts;
  std::vector<int> displs;
  int loc_vec_sz = 0;

  world_.barrier();

  if (size == 1) {
    std::ranges::fill(c_, 0);
    LiterallyMult(a_, b_, c_, a_c_, a_r_, b_c_);
    return true;
  }

  broadcast(world_, a_c_, 0);
  broadcast(world_, b_c_, 0);
  broadcast(world_, b_r_, 0);

  b_.resize(b_r_ * b_c_);
  broadcast(world_, b_.data(), b_c_ * b_r_, 0);

  if (id == 0) {
    send_counts.resize(size);
    displs.resize(size);

    int base_rows = a_r_ / size;
    int remainder = a_r_ % size;

    for (int i = 0; i < size; ++i) {
      send_counts[i] = (base_rows + (i < remainder ? 1 : 0)) * a_c_;
    }

    displs[0] = 0;
    for (int i = 1; i < size; ++i) {
      displs[i] = displs[i - 1] + send_counts[i - 1];
    }
  }

  scatter(world_, send_counts, loc_vec_sz, 0);
  local_data.resize(loc_vec_sz);
  scatterv(world_, a_.data(), send_counts, displs, local_data.data(), loc_vec_sz, 0);

  auto loc_rows_number = loc_vec_sz / a_c_;
  c_.resize(loc_rows_number * b_c_);
  std::ranges::fill(c_, 0);

  LiterallyMult(local_data, b_, c_, a_c_, loc_rows_number, b_c_);

  std::vector<int> gathered_c;
  if (id == 0) {
    gathered_c.resize(a_r_ * b_c_);

    for (int i = 0; i < size; ++i) {
      send_counts[i] = (send_counts[i] / a_c_) * b_c_;
      displs[i] = (displs[i] / a_c_) * b_c_;
    }
  }

  gatherv(world_, c_.data(), loc_rows_number * b_c_, gathered_c.data(), send_counts, displs, 0);

  if (id == 0) {
    c_ = std::move(gathered_c);
  }
  return true;
}

bool RibbonHorSchemeOnlyMatA::PostProcessingImpl() {
  if (world_.rank() == 0) {
    std::ranges::copy(c_, reinterpret_cast<int*>(task_data->outputs[0]));
  }
  return true;
}
}  // namespace somov_i_ribbon_hor_scheme_only_mat_a_mpi