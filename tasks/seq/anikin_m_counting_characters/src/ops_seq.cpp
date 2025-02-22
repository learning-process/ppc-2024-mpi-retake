// Anikin Maksim 2025
#include "seq/anikin_m_counting_characters/include/ops_seq.hpp"

#include <cmath>
#include <vector>
#include <random>

void anikin_m_counting_characters_seq::create_data_vector(std::vector<char> *invec, std::string str) {
  for (auto a : str) {
    invec->push_back(a);
  }
}
void anikin_m_counting_characters_seq::create_randdata_vector(std::vector<char>* invec, int count) {
  for (int i = 0; i < count; i++) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis('A', 'Z');
    char randomChar = static_cast<char>(dis(gen));
    invec->push_back(randomChar);
  }
}

bool anikin_m_counting_characters_seq::TestTaskSequential::ValidationImpl() {
  return (task_data->inputs.size() == 2) && 
         (task_data->inputs_count.size() == 2) &&
         (task_data->outputs.size() == 1);
}

bool anikin_m_counting_characters_seq::TestTaskSequential::PreProcessingImpl() {
  int input1_size = task_data->inputs_count[0];
  int input2_size = task_data->inputs_count[1];

  res = input1_size - input2_size;
  
  if (res <= 0) {
    auto *inlarge_ptr = reinterpret_cast<char *>(task_data->inputs[1]);
    input_1 = std::vector<char>(inlarge_ptr, inlarge_ptr + input2_size);

    auto *insmall_ptr = reinterpret_cast<char *>(task_data->inputs[0]);
    input_2 = std::vector<char>(insmall_ptr, insmall_ptr + input1_size);

    res = abs(res);
  } else {
    auto *inlarge_ptr = reinterpret_cast<char *>(task_data->inputs[0]);
    input_1 = std::vector<char>(inlarge_ptr, inlarge_ptr + input1_size);

    auto *insmall_ptr = reinterpret_cast<char *>(task_data->inputs[1]);
    input_2 = std::vector<char>(insmall_ptr, insmall_ptr + input2_size);
  }
  return true;
}

bool anikin_m_counting_characters_seq::TestTaskSequential::RunImpl() { 
  auto b = input_1.begin();
  for (auto a : input_2) {
    if ((a) != (*b)) res++;
    b++;
  }
  return true;
}

bool anikin_m_counting_characters_seq::TestTaskSequential::PostProcessingImpl() { 
  reinterpret_cast<int *>(task_data->outputs[0])[0] = res;
  return true;
}