#include "seq/kabalova_v_strongin/include/strongin.h"

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

bool kabalova_v_strongin_seq::TestTaskSequential::PreProcessingImpl() {
  result_.first = 0;
  result_.second = 0;
  auto* input_data1 = reinterpret_cast<double*>(task_data->inputs[0]);
  std::copy(input_data1, input_data1 + 1, &left_);
  auto* input_data2 = reinterpret_cast<double*>(task_data->inputs[1]);
  std::copy(input_data2, input_data2 + 1, &right_);
  return true;
}

bool kabalova_v_strongin_seq::TestTaskSequential::ValidationImpl() {
  return task_data->inputs_count[0] == 2 && task_data->outputs_count[0] == 2 && task_data->inputs.size() == 2 &&
         task_data->outputs.size() == 2;
}

bool kabalova_v_strongin_seq::TestTaskSequential::RunImpl() {
  std::vector<std::pair<double, double>> v;
  v.emplace_back(left_, f_(left_));
  v.emplace_back(right_, f_(right_));

  double eps = 0.0001;
  double lipsh = 0.0;
  double r = 2.0;
  int k = 2;
  int s = 0;
  while (true) {
    // Шаг 1. Вычисление константы Липшица.
    for (int i = 0; i < (k - 1); ++i) {
      double new_lipsh = std::abs((v[i + 1].second - v[i].second) / (v[i + 1].first - v[i].first));
      lipsh = std::max(new_lipsh, lipsh);
    }
    double m = 1.0;
    if (lipsh != 0) {
      m = r * lipsh;
    }
    // Шаг 2. Вычисление характеристики.
    s = 0;
    // Самое первое вычисление характеристики.
    double ch = (m * (v[1].first - v[0].first)) +
                ((v[1].second - v[0].second) * (v[1].second - v[0].second) / (m * (v[1].first - v[0].first))) -
                (2 * (v[1].second + v[0].second));
    // Последующие вычисления характеристик, поиск максимальной.
    for (int i = 1; i < (k - 1); ++i) {
      double new_ch =
          (m * (v[i + 1].first - v[i].first)) +
          ((v[i + 1].second - v[i].second) * (v[i + 1].second - v[i].second) / (m * (v[i + 1].first - v[i].first))) -
          (2 * (v[i + 1].second + v[i].second));
      if (new_ch > ch) {
        // Как только нашли - обновили интервал, чтобы найти точку на интервале максимальной характеристики
        s = i;
        ch = new_ch;
      }
    }
    // Шаг 3. Новая точка разбиения на интервале максимальной характеристики.
    double new_x = ((v[s].first + v[s + 1].first) / 2) - ((v[s + 1].second - v[s].second) / (2 * m));
    std::pair<double, double> new_point = std::pair<double, double>(new_x, f_(new_x));
    // Шаг 4. Проверка критерия останова по точности.
    if ((v[s + 1].first - v[s].first) <= eps) {
      result_ = v[s + 1];
      break;
    }
    // Иначе - упорядочиваем массив по возрастания и возвращаемся на шаг 1.
    v.emplace_back(new_point);
    std::ranges::sort(v);
    k++;
  }
  return true;
}

bool kabalova_v_strongin_seq::TestTaskSequential::PostProcessingImpl() {
  reinterpret_cast<double*>(task_data->outputs[0])[0] = result_.first;
  reinterpret_cast<double*>(task_data->outputs[1])[0] = result_.second;
  return true;
}
