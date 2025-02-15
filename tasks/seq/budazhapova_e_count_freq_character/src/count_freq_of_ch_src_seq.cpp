#include "seq/budazhapova_e_count_freq_character/include/count_freq_counter_header.h"

using namespace std::chrono_literals;

bool budazhapova_e_count_freq_chart_seq::TestTaskSequential::PreProcessingImpl() {
  input_ = *reinterpret_cast<std::string*>(task_data->inputs[0]);
  symb = input_[0];
  res = 0;
  return true;
}

bool budazhapova_e_count_freq_chart_seq::TestTaskSequential::ValidationImpl() {
  return task_data->inputs_count[0] == 1;
  // Или, если вы хотите убедиться, что inputs_count[0] больше 0:
  // return task_data->inputs_count[0] > 0;
}

bool budazhapova_e_count_freq_counter_seq::TestTaskSequential::RunImpl() {
  for (unsigned long i = 0; i < input_.length(); i++) {
    if (input_[i] == symb) {
      res++;
    }
  }
  return true;
}

bool budazhapova_e_count_freq_chart_seq::TestTaskSequential::PostProcessingImpl() {
  reinterpret_cast<int*>(task_data->outputs[0])[0] = res;
  return true;
}