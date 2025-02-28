#include "seq/vinyaikina_e_max_of_vector_elements/include/ops_seq.hpp"

#include <cstdint>
#include <limits>
#include <memory>

namespace vinyaikina_e_max_of_vector_elements_seq {

bool VectorMaxSeq::ValidationImpl() { return !task_data->outputs.empty() && task_data->outputs_count[0] == 1; }

bool VectorMaxSeq::PreProcessingImpl() {
  auto* input_ptr = reinterpret_cast<int32_t*>(task_data->inputs[0]);
  input_.resize(task_data->inputs_count[0]);
  std::copy(input_ptr, input_ptr + task_data->inputs_count[0], input_.begin());

  return true;
}

bool VectorMaxSeq::RunImpl() {
  if (input_.empty()) {
    return true;
  }

  max_ = input_[0];
  for (int32_t num : input_) {
    max_ = std::max(num, max_);
  }

  return true;
}

bool VectorMaxSeq::PostProcessingImpl() {
  *reinterpret_cast<int32_t*>(task_data->outputs[0]) = max_;
  return true;
}

}  // namespace vinyaikina_e_max_of_vector_elements_seq