#include "seq/makhov_m_monte_carlo_method/include/ops_seq.hpp"

#include <cmath>
#include <cstddef>
#include <memory>
#include <random>
#include <vector>

bool makhov_m_monte_carlo_method_seq::TestTaskSequential::PreProcessingImpl() {
  func = *reinterpret_cast<std::function<double(const std::vector<double>&)>*>(task_data->inputs[0]);
  numSamples = *reinterpret_cast<int*>(task_data->inputs[1]);

  if (task_data->inputs[2] == nullptr) {
    throw std::runtime_error("task_data->inputs[2] is null");
  }

  auto limits_ptr = reinterpret_cast<std::pair<double, double>*>(task_data->inputs[2]);
  std::uint32_t dimension = task_data->inputs_count[2];
  limits.assign(limits_ptr, limits_ptr + dimension);
  return true;
}

bool makhov_m_monte_carlo_method_seq::TestTaskSequential::ValidationImpl() {
  return task_data->inputs_count.size() == 3 && task_data->outputs_count.size() == 1 &&
         task_data->outputs_count[0] == task_data->inputs_count[0];
}

bool makhov_m_monte_carlo_method_seq::TestTaskSequential::RunImpl() {
  std::random_device rd;
  std::mt19937 gen(rd());

  // Вектор распределений для каждой переменной
  std::vector<std::uniform_real_distribution<>> distributions;
  for (const auto& limit : limits) {
    distributions.emplace_back(limit.first, limit.second);
  }
  double sum = 0.0;

  // Генерация случайных точек и вычисление суммы значений функции
  for (int i = 0; i < numSamples; ++i) {
    std::vector<double> point;
    for (size_t j = 0; j < limits.size(); ++j) {
      point.push_back(distributions[j](gen));  // Генерация случайной точки
    }
    sum += func(point);  // Добавляем значение функции в сумму
  }

  // Вычисление объема области интегрирования
  double volume = 1.0;
  for (const auto& limit : limits) {
    volume *= (limit.second - limit.first);  // Объем = произведение длин интервалов
  }

  // Оценка интеграла
  answer = volume * (sum / numSamples);  // Интеграл = объем * среднее значение
  answerDataPtr = new uint8_t[sizeof(double)];
  std::memcpy(answerDataPtr, &answer, sizeof(double));
  return true;
}

bool makhov_m_monte_carlo_method_seq::TestTaskSequential::PostProcessingImpl() {
  task_data->outputs_count = task_data->inputs_count;
  task_data->outputs[0] = answerDataPtr;
  delete[] answerDataPtr;
  return true;
}
