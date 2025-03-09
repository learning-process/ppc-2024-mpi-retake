// Copyright 2023 Nesterov Alexander
#include "mpi/makhov_m_monte_carlo_method/include/ops_mpi.hpp"

//#include <algorithm>
#include <boost/serialization/utility.hpp>
#include <boost/serialization/vector.hpp>
#include <cstddef>
#include <cstdint>
#include <map>
#include <random>
#include <regex>
#include <string>
#include <vector>

#include "simple_parser.hpp"

bool makhov_m_monte_carlo_method_mpi::TestMPITaskParallel::PreProcessingImpl() {
  if (world_.rank() == 0) {
    funcStr = *reinterpret_cast<std::string*>(task_data->inputs[0]);
    numSamples = *reinterpret_cast<int*>(task_data->inputs[1]);

    auto limitsPtr = reinterpret_cast<std::pair<double, double>*>(task_data->inputs[2]);
    std::uint32_t dimension = task_data->inputs_count[2];
    limits.assign(limitsPtr, limitsPtr + dimension);
  }
  return true;
}

bool makhov_m_monte_carlo_method_mpi::TestMPITaskParallel::ValidationImpl() {
  if (world_.rank() == 0) {
    return task_data->inputs_count.size() == 3 && task_data->outputs_count.size() == 1 &&
           task_data->outputs_count[0] == task_data->inputs_count[0];
  }
  return true;
}

bool makhov_m_monte_carlo_method_mpi::TestMPITaskParallel::RunImpl() {
  boost::mpi::broadcast(world_, numSamples, 0);
  boost::mpi::broadcast(world_, limits, 0);
  boost::mpi::broadcast(world_, funcStr, 0);

  std::regex varRegex("[a-z]");  // Регулярное выражение для переменных (a-z)
  std::smatch matches;
  std::string::const_iterator searchStart(funcStr.cbegin());
  std::vector<std::string> variables;

  while (std::regex_search(searchStart, funcStr.cend(), matches, varRegex)) {
    variables.push_back(matches.str(0));
    searchStart = matches.suffix().first;
  }

  // Удаляем дубликаты
  std::sort(variables.begin(), variables.end());
  variables.erase(std::unique(variables.begin(), variables.end()), variables.end());

  // Контейнер для хранения переменных и их значений
  std::map<std::string, double> varValues;

  // Добавляем переменные в таблицу символов
  for (const auto& var : variables) {
    varValues[var] = 0.0;  // Инициализация переменных нулями
  }

  std::random_device rd;
  std::mt19937 gen(rd() + world_.rank());  // Уникальный seed для каждого процесса
  std::uniform_real_distribution<> dis(limits[0].first, limits[0].second);

  // Вычисление количества точек для каждого процесса
  int localSamples = numSamples / world_.size();
  if (world_.rank() == world_.size() - 1) {
    localSamples += numSamples % world_.size();
  }

  double localSum = 0.0;
  for (int i = 0; i < localSamples; ++i) {  // Генерация случайной точки
    // Генерация случайных значений для переменных
    for (auto& var : varValues) {
      var.second = dis(gen);  // Случайное значение переменной
    }

    // Вычисление функции
    SimpleParser parser(funcStr, varValues);
    localSum += parser.parse();
  }

  if (world_.rank() == 0) {
    globalSum = localSum;

    // Принимаем данные от всех остальных процессов
    for (int i = 1; i < world_.size(); ++i) {
      double receivedSum;
      world_.recv(i, 0, receivedSum);  // Принимаем данные от процесса i
      globalSum += receivedSum;        // Суммируем
    }
  } else {
    // Остальные процессы отправляют свои локальные суммы на корневой процесс
    world_.send(0, 0, localSum);  // Отправляем данные на процесс 0
  }

  if (world_.rank() == 0) {
    double volume = 1.0;
    for (const auto& limit : limits) {
      volume *= (limit.second - limit.first);
    }
    answer = volume * (globalSum / numSamples);
    answerDataPtr = new uint8_t[sizeof(double)];
    std::memcpy(answerDataPtr, &answer, sizeof(double));
  }
  world_.barrier();
  return true;
}

bool makhov_m_monte_carlo_method_mpi::TestMPITaskParallel::PostProcessingImpl() {
  if (world_.rank() == 0) {
    task_data->outputs_count = task_data->inputs_count;
    task_data->outputs[0] = answerDataPtr;
    delete[] answerDataPtr;
  }
  return true;
}
