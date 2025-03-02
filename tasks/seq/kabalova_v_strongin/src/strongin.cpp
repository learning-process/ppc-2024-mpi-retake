#include "seq/kabalova_v_strongin/include/strongin.h"

#include <random>
#include <thread>

bool kabalova_v_strongin_seq::TestTaskSequential::PreProcessingImpl() {
  result.first = 0;
  result.second = 0;
  auto* inputData1 = reinterpret_cast<double*>(task_data->inputs[0]);
  std::copy(inputData1, inputData1 + 1, &left);
  auto* inputData2 = reinterpret_cast<double*>(task_data->inputs[1]);
  std::copy(inputData2, inputData2 + 1, &right);
  return true;
}

bool kabalova_v_strongin_seq::TestTaskSequential::ValidationImpl() {
  return task_data->inputs_count[0] == 2 && task_data->outputs_count[0] == 2 && task_data->inputs.size() == 2 &&
         task_data->outputs.size() == 2;
}

bool kabalova_v_strongin_seq::TestTaskSequential::RunImpl() {
  std::vector<std::pair<double, double>> v;
  v.push_back(std::pair<double, double>(left, f(left)));
  v.push_back(std::pair<double, double>(right, f(right)));

  double eps = 0.0001;
  double M = 0.0;
  double r = 2.0;
  int k = 2;
  int s = 0;
  while (true) {
    // Шаг 1. Вычисление константы Липшица.
    for (int i = 0; i < (k - 1); ++i) {
      double newM = std::abs((v[i + 1].second - v[i].second) / (v[i + 1].first - v[i].first));
      M = std::max(newM, M);
    }
    double m = 1.0;
    if (M != 0) m = r * M;
    // Шаг 2. Вычисление характеристики.
    s = 0;
    // Самое первое вычисление характеристики.
    double R = m * (v[1].first - v[0].first) +
               (v[1].second - v[0].second) * (v[1].second - v[0].second) / (m * (v[1].first - v[0].first)) -
               2 * (v[1].second + v[0].second);
    // Последующие вычисления характеристик, поиск максимальной.
    for (int i = 1; i < (k - 1); ++i) {
      double newR =
          m * (v[i + 1].first - v[i].first) +
          (v[i + 1].second - v[i].second) * (v[i + 1].second - v[i].second) / (m * (v[i + 1].first - v[i].first)) -
          2 * (v[i + 1].second + v[i].second);
      if (newR > R) {
        // Как только нашли - обновили интервал, чтобы найти точку на интервале максимальной характеристики
        s = i;
        R = newR;
      }
    }
    // Шаг 3. Новая точка разбиения на интервале максимальной характеристики.
    double newX = (v[s].first + v[s + 1].first) / 2 - (v[s + 1].second - v[s].second) / (2 * m);
    std::pair<double, double> newPoint = std::pair<double, double>(newX, f(newX));
    // Шаг 4. Проверка критерия останова по точности.
    if ((v[s + 1].first - v[s].first) <= eps) {
      result = v[s + 1];
      break;
    }
    // Иначе - упорядочиваем массив по возрастания и возвращаемся на шаг 1.
    v.push_back(newPoint);
    std::sort(v.begin(), v.end());
    k++;
  }
  return true;
}

bool kabalova_v_strongin_seq::TestTaskSequential::PostProcessingImpl() {
  reinterpret_cast<double*>(task_data->outputs[0])[0] = result.first;
  reinterpret_cast<double*>(task_data->outputs[1])[0] = result.second;
  return true;
}
