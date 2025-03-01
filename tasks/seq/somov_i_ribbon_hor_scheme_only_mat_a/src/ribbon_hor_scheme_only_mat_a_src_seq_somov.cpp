#include <algorithm>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>

#include "seq/somov_i_ribbon_hor_scheme_only_mat_a/include/ribbon_hor_scheme_only_mat_a_header_seq_somov.hpp"
namespace somov_i_ribbon_hor_scheme_only_mat_a_seq {
void GetRndVector(std::vector<int>& vec) {
  std::random_device rd;
  std::default_random_engine reng(rd());
  std::uniform_int_distribution<int> dist(-static_cast<int>(vec.size()) - 1, static_cast<int>(vec.size()) - 1);
  std::ranges::generate(vec, [&dist, &reng] { return dist(reng); });
}
void ClearMult(const std::vector<int>& a, const std::vector<int>& b, std::vector<int>& c, int a_c_, int a_r_,
               int b_c_) {
  for (size_t i = 0; i < a_r_; ++i) {
    for (size_t j = 0; j < b_c_; ++j) {
      for (size_t p = 0; p < a_c_; ++p) {
        c[i * b_c_ + j] += a[i * a_c_ + p] * b[p * b_c_ + j];
      }
    }
  }
}
bool RibbonHorSchemeOnlyMatA::PreProcessingImpl() {
  a_c_ = static_cast<int>(task_data->inputs_count[0]);
  a_r_ = static_cast<int>(task_data->inputs_count[1]);
  b_c_ = static_cast<int>(task_data->inputs_count[2]);
  b_r_ = static_cast<int>(task_data->inputs_count[3]);

  a_.resize(a_c_ * a_r_);
  b_.resize(b_c_ * b_r_);
  c_.resize(a_r_ * b_c_);

  std::ranges::copy(reinterpret_cast<int*>(task_data->inputs[0]),
                    reinterpret_cast<int*>(task_data->inputs[0]) + a_c_ * a_r_, a_.begin());

  std::ranges::copy(reinterpret_cast<int*>(task_data->inputs[1]),
                    reinterpret_cast<int*>(task_data->inputs[1]) + b_c_ * b_r_, b_.begin());

  return true;
}

bool RibbonHorSchemeOnlyMatA::ValidationImpl() {
  return (task_data->outputs_count[0] > 0 &&
          static_cast<int>(task_data->inputs_count[0]) == static_cast<int>(task_data->inputs_count[3]));
}

bool RibbonHorSchemeOnlyMatA::RunImpl() {
  std::ranges::fill(c_, 0);
  for (size_t i = 0; i < a_r_; ++i) {
    for (size_t j = 0; j < b_c_; ++j) {
      for (size_t p = 0; p < a_c_; ++p) {
        c_[i * b_c_ + j] += a_[i * a_c_ + p] * b_[p * b_c_ + j];
      }
    }
  }
  return true;
}

bool RibbonHorSchemeOnlyMatA::PostProcessingImpl() {
  std::ranges::copy(c_, reinterpret_cast<int*>(task_data->outputs[0]));
  return true;
}
}  // namespace somov_i_ribbon_hor_scheme_only_mat_a_seq