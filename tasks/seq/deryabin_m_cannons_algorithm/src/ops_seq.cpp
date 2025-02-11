#include "seq/deryabin_m_cannons_algorithm/include/ops_seq.hpp"

#include <thread>

bool deryabin_m_cannons_algorithm_seq::CannonsAlgorithmTaskSequential::PreProcessingImpl() {
  input_matrix_A = reinterpret_cast<std::vector<double> *>(taskData->inputs[0])[0];
  input_matrix_B = reinterpret_cast<std::vector<double> *>(taskData->inputs[1])[0];
  output_matrix_C = std::vector<double>(input_matrix_A.size());
  return true;
}

bool deryabin_m_cannons_algorithm_seq::CannonsAlgorithmTaskSequential::ValidationImpl() {
  return taskData->inputs_count[0] == taskData->inputs_count[1] &&
         taskData->inputs_count[1] == pow((unsigned short)sqrt(taskData->inputs_count[0]), 2) &&
         taskData->outputs_count[0] == 1;
}

bool deryabin_m_cannons_algorithm_seq::CannonsAlgorithmTaskSequential::RunImpl() {
  unsigned short i = 0;
  unsigned short j;
  unsigned short count;
  auto dimension = (unsigned short)sqrt(input_matrix_A.size());
  while (i != dimension) {
    j = 0;
    while (j != dimension) {
      count = 0;
      while (count != dimension) {
        output_matrix_C[i * dimension + j] +=
            input_matrix_A[i * dimension + count] * input_matrix_B[count * dimension + j];
        count++;
      }
      j++;
    }
    i++;
  }
  return true;
}

bool deryabin_m_cannons_algorithm_seq::CannonsAlgorithmTaskSequential::PostProcessingImpl() {
  reinterpret_cast<std::vector<double> *>(taskData->outputs[0])[0] = output_matrix_C;
  return true;
}
