#include <algorithm>
#include <cstring>
#include <random>
#include <vector>

#include "seq/somov_i_num_of_alternations_signs/include/num_of_alternations_signs_header_seq_somov.hpp"
namespace somov_i_num_of_alternations_signs_seq {
void CheckForAlternationSigns(const std::vector<int>& vec, int& out) {
  out = 0;
  for (int i = 0; i < static_cast<int>(vec.size()) - 1; ++i) {
    if (vec[i] * vec[i + 1] < 0) {
      ++out;
    }
  }
}
void GetRndVector(std::vector<int>& vec) {
  std::random_device rd;
  std::default_random_engine reng(rd());
  std::uniform_int_distribution<int> dist(-static_cast<int>(vec.size()) - 1, static_cast<int>(vec.size()) - 1);
  std::ranges::generate(vec, [&dist, &reng] { return dist(reng); });
}
bool NumOfAlternationsSigns::PreProcessingImpl() {
  // Init vectors
  sz_ = static_cast<int>(task_data->inputs_count[0]);
  input_ = std::vector<int>(sz_);
  std::ranges::copy(reinterpret_cast<int*>(task_data->inputs[0]), reinterpret_cast<int*>(task_data->inputs[0]) + sz_,
                    input_.begin());
  return true;
}

bool NumOfAlternationsSigns::ValidationImpl() { return (task_data->outputs_count[0] > 0); }

bool NumOfAlternationsSigns::RunImpl() {
  output_ = 0;

  for (int i = 0; i < sz_ - 1; ++i) {
    if (input_[i] * input_[i + 1] < 0) {
      ++output_;
    }
  }
  return true;
}

bool NumOfAlternationsSigns::PostProcessingImpl() {
  reinterpret_cast<int*>(task_data->outputs[0])[0] = output_;
  return true;
}
}  // namespace somov_i_num_of_alternations_signs_seq