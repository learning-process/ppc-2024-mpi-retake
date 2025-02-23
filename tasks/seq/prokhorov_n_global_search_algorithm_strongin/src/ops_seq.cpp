// Copyright 2024 Nesterov Alexander
#include "seq/prokhorov_n_global_search_algorithm_strongin/include/ops_seq.hpp"

#include <cmath>
#include <thread>

using namespace std::chrono_literals;

namespace prokhorov_n_global_search_algorithm_strongin_seq {

bool prokhorov_n_global_search_algorithm_strongin_seq::TestTaskSequential::PreProcessingImpl() {
  a = reinterpret_cast<double*>(task_data->inputs[0])[0];
  b = reinterpret_cast<double*>(task_data->inputs[1])[0];
  epsilon = reinterpret_cast<double*>(task_data->inputs[2])[0];

  f = [](double x) { return x * x; };

  result = 0.0;
  return true;
}

bool prokhorov_n_global_search_algorithm_strongin_seq::TestTaskSequential::ValidationImpl() {
  return task_data->inputs_count[0] == 1 && task_data->inputs_count[1] == 1 && task_data->inputs_count[2] == 1 &&
         task_data->outputs_count[0] == 1;
}

bool prokhorov_n_global_search_algorithm_strongin_seq::TestTaskSequential::RunImpl() {
  result = stronginAlgorithm();
  std::this_thread::sleep_for(20ms);
  return true;
}

bool prokhorov_n_global_search_algorithm_strongin_seq::TestTaskSequential::PostProcessingImpl() {
  reinterpret_cast<double*>(task_data->outputs[0])[0] = result;
  return true;
}

double prokhorov_n_global_search_algorithm_strongin_seq::TestTaskSequential::stronginAlgorithm() {
  double x_min = a;
  double f_min = f(x_min);

  while ((b - a) > epsilon) {
    double x1 = a + (b - a) / 3.0;
    double x2 = b - (b - a) / 3.0;

    double f1 = f(x1);
    double f2 = f(x2);

    if (f1 < f2) {
      b = x2;
      if (f1 < f_min) {
        f_min = f1;
        x_min = x1;
      }
    } else {
      a = x1;
      if (f2 < f_min) {
        f_min = f2;
        x_min = x2;
      }
    }
  }

  return x_min;
}
}  // namespace prokhorov_n_global_search_algorithm_strongin_seq