// Copyright 2024 Nesterov Alexander
#include "seq/prokhorov_n_global_search_algorithm_strongin/include/ops_seq.hpp"

#include <chrono>
#include <cmath>
#include <thread>

using namespace std::chrono_literals;

namespace prokhorov_n_global_search_algorithm_strongin_seq {

bool prokhorov_n_global_search_algorithm_strongin_seq::TestTaskSequential::PreProcessingImpl() {
  a_ = reinterpret_cast<double*>(task_data->inputs[0])[0];
  b_ = reinterpret_cast<double*>(task_data->inputs[1])[0];
  epsilon_ = reinterpret_cast<double*>(task_data->inputs[2])[0];

  f_ = [](double x) { return x * x; };

  result_ = 0.0;
  return true;
}

bool prokhorov_n_global_search_algorithm_strongin_seq::TestTaskSequential::ValidationImpl() {
  return task_data->inputs_count[0] == 1 && task_data->inputs_count[1] == 1 && task_data->inputs_count[2] == 1 &&
         task_data->outputs_count[0] == 1;
}

bool prokhorov_n_global_search_algorithm_strongin_seq::TestTaskSequential::RunImpl() {
  result_ = StronginAlgorithm();
  std::this_thread::sleep_for(std::chrono::milliseconds(20)); 
  return true;
}

bool prokhorov_n_global_search_algorithm_strongin_seq::TestTaskSequential::PostProcessingImpl() {
  reinterpret_cast<double*>(task_data->outputs[0])[0] = result_;
  return true;
}

double prokhorov_n_global_search_algorithm_strongin_seq::TestTaskSequential::StronginAlgorithm() {
  double x_min = a_;
  double f_min = f_(x_min);

  while ((b_ - a_) > epsilon_) {
    double x1 = a_ + ((b_ - a_) / 3.0);
    double x2 = b_ - ((b_ - a_) / 3.0);

    double f1 = f_(x1);
    double f2 = f_(x2);

    if (f1 < f2) {
      b_ = x2;
      if (f1 < f_min) {
        f_min = f1;
        x_min = x1;
      }
    } else {
      a_ = x1;
      if (f2 < f_min) {
        f_min = f2;
        x_min = x2;
      }
    }
  }

  return x_min;
}
}  // namespace prokhorov_n_global_search_algorithm_strongin_seq