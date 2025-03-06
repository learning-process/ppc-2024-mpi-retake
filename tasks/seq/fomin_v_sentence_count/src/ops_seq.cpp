#include "seq/fomin_v_sentence_count/include/ops_seq.hpp"

#include <thread>

using namespace std::chrono_literals;

bool fomin_v_sentence_count::SentenceCountSequential::PreProcessingImpl() {
  // Получаем входную строку
  input_ = std::string(std::move(reinterpret_cast<char *>(task_data->inputs[0])));
  sentence_count = 0;
  return true;
}

bool fomin_v_sentence_count::SentenceCountSequential::ValidationImpl() {
  // Проверяем, что входные данные содержат строку
  return task_data->inputs_count.size() == 1 && task_data->outputs_count.size() == 1 &&
         task_data->outputs_count[0] == 1;
}

bool fomin_v_sentence_count::SentenceCountSequential::RunImpl() {
  // Подсчитываем количество предложений
  for (int i = 0; input_[i] != '\0'; ++i) {
    if (ispunct(input_[i]) && (input_[i] == '.' || input_[i] == '!' || input_[i] == '?')) {
      sentence_count++;
    }
  }
  return true;
}

bool fomin_v_sentence_count::SentenceCountSequential::PostProcessingImpl() {
  // Записываем результат в выходные данные
  reinterpret_cast<int *>(task_data->outputs[0])[0] = sentence_count;
  return true;
}