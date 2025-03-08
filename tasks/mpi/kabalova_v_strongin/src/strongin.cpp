// Copyright_ 2024 Kabalova Valeria
#include "mpi/kabalova_v_strongin/include/strongin.h"

#include <algorithm>
#include <boost/mpi/collectives/broadcast.hpp>
#include <boost/mpi/collectives/gather.hpp>
#include <cmath>
#include <functional>
#include <utility>
#include <vector>

bool kabalova_v_strongin_mpi::TestMPITaskSequential::PreProcessingImpl() {
  result_.first = 0;
  result_.second = 0;
  auto* input_data1 = reinterpret_cast<double*>(task_data->inputs[0]);
  std::copy(input_data1, input_data1 + 1, &left_);
  auto* input_data2 = reinterpret_cast<double*>(task_data->inputs[1]);
  std::copy(input_data2, input_data2 + 1, &right_);
  return true;
}

bool kabalova_v_strongin_mpi::TestMPITaskSequential::ValidationImpl() {
  return task_data->inputs_count[0] == 2 && task_data->outputs_count[0] == 2 && task_data->inputs.size() == 2 &&
         task_data->outputs.size() == 2;
}

bool kabalova_v_strongin_mpi::TestMPITaskSequential::RunImpl() {
  std::vector<std::pair<double, double>> v;
  v.emplace_back(left_, f_(&left_));
  v.emplace_back(right_, f_(&right_));

  double eps = 0.0001;
  double lipsh = 0.0;
  double r = 2.0;
  int k = 2;
  int s = 0;
  while (true) {
    // Шаг 1. Вычисление константы Липшица.
    for (int i = 0; i < (k - 1); i++) {
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
    for (int i = 1; i < (k - 1); i++) {
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
    std::pair<double, double> new_point = std::pair<double, double>(new_x, f_(&new_x));
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

bool kabalova_v_strongin_mpi::TestMPITaskSequential::PostProcessingImpl() {
  reinterpret_cast<double*>(task_data->outputs[0])[0] = result_.first;
  reinterpret_cast<double*>(task_data->outputs[1])[0] = result_.second;
  return true;
}

bool kabalova_v_strongin_mpi::TestMPITaskParallel::PreProcessingImpl() {
  if (world_.rank() == 0) {
    result_.first = 0;
    result_.second = 0;
    auto* input_data1 = reinterpret_cast<double*>(task_data->inputs[0]);
    std::copy(input_data1, input_data1 + 1, &left_);
    auto* input_data2 = reinterpret_cast<double*>(task_data->inputs[1]);
    std::copy(input_data2, input_data2 + 1, &right_);
    return true;
  }
  return true;
}

bool kabalova_v_strongin_mpi::TestMPITaskParallel::ValidationImpl() {
  bool flag = true;
  if (world_.rank() == 0) {
    flag = task_data->inputs_count[0] == 2 && task_data->outputs_count[0] == 2 && task_data->inputs.size() == 2 &&
           task_data->outputs.size() == 2;
  }
  broadcast(world_, flag, 0);
  return flag;
}

double kabalova_v_strongin_mpi::Algorithm(double left, double right, const std::function<double(double*)>& f,
                                          double eps) {
  std::vector<std::pair<double, double>> v;
  v.emplace_back(left, f(&left));
  v.emplace_back(right, f(&right));
  double lipsh = 0.0;
  double r = 2.0;
  int k = 2;
  int s = 0;
  // Основной цикл
  while (true) {
    for (int i = 0; i < (k - 1); ++i) {
      double new_lipsh = std::abs((v[i + 1].second - v[i].second) / (v[i + 1].first - v[i].first));
      lipsh = std::max(new_lipsh, lipsh);
    }
    double m = 1.0;
    if (lipsh != 0) {
      m = r * lipsh;
    }
    // Вычисление характеристики
    s = 0;
    // Первое вычисление характеристики ch
    double ch = (m * (v[1].first - v[0].first)) +
                ((v[1].second - v[0].second) * (v[1].second - v[0].second) / (m * (v[1].first - v[0].first))) -
                (2 * (v[1].second + v[0].second));
    for (int i = 1; i < (k - 1); i++) {
      double new_ch =
          (m * (v[i + 1].first - v[i].first)) +
          ((v[i + 1].second - v[i].second) * (v[i + 1].second - v[i].second) / (m * (v[i + 1].first - v[i].first))) -
          (2 * (v[i + 1].second + v[i].second));
      if (new_ch > ch) {
        s = i;
        ch = new_ch;
      }
    }
    double new_x = ((v[s].first + v[s + 1].first) / 2) - ((v[s + 1].second - v[s].second) / (2 * m));
    std::pair<double, double> new_point = std::pair<double, double>(new_x, f(&new_x));
    if ((v[s + 1].first - v[s].first) <= eps) {
      return v[s + 1].first;
    }
    v.emplace_back(new_point);
    std::ranges::sort(v);
    k++;
  }
}

bool kabalova_v_strongin_mpi::TestMPITaskParallel::RunImpl() {
  unsigned int size = world_.size();
  unsigned int rank = world_.rank();
  double segment = 0;

  if (size == 1) {
    result_.first = kabalova_v_strongin_mpi::Algorithm(left_, right_, f_, 0.0001);
    result_.second = f_(&result_.first);
    return true;
  }
  std::vector<double> local_answer(size);
  if (rank == 0) {
    segment = std::abs(right_ - left_) / world_.size();
  }
  broadcast(world_, segment, 0);
  broadcast(world_, left_, 0);
  broadcast(world_, right_, 0);
  double local_left = left_ + (segment * rank);
  double local_right = local_left + segment;

  double local_result = kabalova_v_strongin_mpi::Algorithm(local_left, local_right, f_, 0.0001);

  boost::mpi::gather(world_, local_result, local_answer, 0);

  if (rank == 0) {
    double answer = f_(local_answer.data());
    for (unsigned int i = 1; i < size; i++) {
      if (f_(&local_answer[i]) < answer) {
        std::swap(local_answer[0], local_answer[i]);
        answer = f_(&local_answer[i]);
      }
    }
  }
  result_.first = local_answer[0];
  result_.second = f_(local_answer.data());
  return true;
}

bool kabalova_v_strongin_mpi::TestMPITaskParallel::PostProcessingImpl() {
  if (world_.rank() == 0) {
    reinterpret_cast<double*>(task_data->outputs[0])[0] = result_.first;
    reinterpret_cast<double*>(task_data->outputs[1])[0] = result_.second;
  }
  return true;
}