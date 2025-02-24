// Copyright 2024 Nesterov Alexander
#include "seq/shishkarev_a_sum_of_vector_elements/include/ops_seq.hpp"

#include <cstdint>
#include <numeric>

template <class InOutType>
bool shishkarev_a_sum_of_vector_elements_seq::VectorSumSequential<InOutType>::PreProcessingImpl() {
  const auto input_size = task_data->inputs_count[0];
  input_data_ = std::vector<InOutType>(input_size);

  auto input_ptr = reinterpret_cast<InOutType*>(task_data->inputs[0]);
  std::copy(input_ptr, input_ptr + input_size, input_data_.begin());

  result_ = InOutType{};
  return true;
}

template <class InOutType>
bool shishkarev_a_sum_of_vector_elements_seq::VectorSumSequential<InOutType>::ValidationImpl() {
  const bool is_input_valid = task_data->inputs_count[0] > 0;
  const bool is_output_valid = task_data->outputs_count[0] == 1;
  return is_input_valid && is_output_valid;
}

template <class InOutType>
bool shishkarev_a_sum_of_vector_elements_seq::VectorSumSequential<InOutType>::RunImpl() {
  result_ = std::accumulate(input_data_.begin(), input_data_.end(), InOutType{});
  return true;
}

template <class InOutType>
bool shishkarev_a_sum_of_vector_elements_seq::VectorSumSequential<InOutType>::PostProcessingImpl() {
  auto output_ptr = reinterpret_cast<InOutType*>(task_data->outputs[0]);
  output_ptr[0] = result_;
  return true;
}

template class shishkarev_a_sum_of_vector_elements_seq::VectorSumSequential<int32_t>;
template class shishkarev_a_sum_of_vector_elements_seq::VectorSumSequential<double>;
template class shishkarev_a_sum_of_vector_elements_seq::VectorSumSequential<uint8_t>;
template class shishkarev_a_sum_of_vector_elements_seq::VectorSumSequential<int64_t>;
template class shishkarev_a_sum_of_vector_elements_seq::VectorSumSequential<float>;
