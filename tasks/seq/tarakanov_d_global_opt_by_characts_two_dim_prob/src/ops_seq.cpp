// @copyright Tarakanov Denis
#include "seq/tarakanov_d_global_opt_by_characts_two_dim_prob/include/ops_seq.hpp"
#include <limits>

double tarakanov_d_global_opt_two_dim_prob_seq::GetConstraintsSum(double x, double y, int num, std::vector<double> vec) {
    return vec[num * 3] * x + vec[num * 3 + 1] * y - vec[num * 3 + 2];
}

bool tarakanov_d_global_opt_two_dim_prob_seq::CheckConstraints(double x, double y, int constraint_num,
                                                                            std::vector<double> constraints) {
  return GetConstraintsSum(x, y, constraint_num, constraints) <= 0;
}

bool tarakanov_d_global_opt_two_dim_prob_seq::IsAcceptable (double x, double y, int constraint_num, std::vector<double> constraints) {
  for (int i = 0; i < constraint_num; ++i) {
    if (false == CheckConstraints(x, y, i, constraints)) {
      return false;
    }
  }

  return true;
}

double tarakanov_d_global_opt_two_dim_prob_seq::ComputeFunction(double x, double y,
                                                                           std::vector<double> params) {
  return (x - params[0]) * (x - params[0]) + (y - params[1]) * (y - params[1]);
}

bool tarakanov_d_global_opt_two_dim_prob_seq::GlobalOptSequential::ValidationImpl() {
  bool rc = (1 == task_data->outputs_count[0]);
  rc = rc && (2 == task_data->inputs_count.size());
  return rc;
}

bool tarakanov_d_global_opt_two_dim_prob_seq::GlobalOptSequential::PreProcessingImpl() {
  bounds.push_back(reinterpret_cast<double*>(task_data->inputs[0])[0]);
  bounds.push_back(reinterpret_cast<double*>(task_data->inputs[0])[1]);
  bounds.push_back(reinterpret_cast<double*>(task_data->inputs[0])[2]);
  bounds.push_back(reinterpret_cast<double*>(task_data->inputs[0])[3]);

  num_constraints = task_data->inputs_count[0];
  constraints.resize(num_constraints * 3);
  mode = task_data->inputs_count[1];

  func_params.push_back(reinterpret_cast<double*>(task_data->inputs[1])[0]);
  func_params.push_back(reinterpret_cast<double*>(task_data->inputs[1])[1]);

  for (int i = 0; i < num_constraints * 3; i++) {
    constraints[i] = (reinterpret_cast<double*>(task_data->inputs[2])[i]);
  }

  step = *reinterpret_cast<double*>(task_data->inputs[3]);
  
  return true;
}

bool tarakanov_d_global_opt_two_dim_prob_seq::GlobalOptSequential::RunImpl() {
  switch (mode) {
  case 1:
    result = std::numeric_limits<double>::min();
    break;
  case 0:
    result = std::numeric_limits<double>::max();
    break;
  }

  double accuracy = 1e-6;
  double last_result = std::numeric_limits<double>::max();
  auto factor = static_cast<int>(1.0 / step);

  for (; step >= accuracy; step /= 2.0) {
    double local_minX = bounds[0];
    double local_minY = bounds[2];

    auto int_minX = static_cast<int>(bounds[0] * factor);
    auto x = int_minX;
    auto int_maxX = static_cast<int>(bounds[1] * factor);
    
    auto int_minY = static_cast<int>(bounds[2] * factor);
    auto y = int_minY;
    auto int_maxY = static_cast<int>(bounds[3] * factor);

    while (x < int_maxX) {
      auto real_x = static_cast<double>(++x) / factor;
      while (y < int_maxY) {
        auto real_y = static_cast<double>(++y) / factor;         
        
        if (true == IsAcceptable(real_x, real_y, num_constraints, constraints)) {
          auto value = ComputeFunction(real_x, real_y, func_params);
          switch (mode) {
            case 0:
              if (value < result) {
                result = value;
                local_minX = real_x;
                local_minY = real_y;
              }

              break;
            case 1:
              if (value > result) {
                result = value;
                local_minX = real_x;
                local_minY = real_y;
              }

              break;
          }
        }
      }
    }

    if (std::abs(last_result - result) < accuracy) {
      break;
    }

    bounds[0] = std::max(local_minX - 2 * step, bounds[0]);
    bounds[1] = std::min(local_minX + 2 * step, bounds[1]);
    bounds[2] = std::max(local_minY - 2 * step, bounds[2]);
    bounds[3] = std::min(local_minY + 2 * step, bounds[3]);

    last_result = result;
  }

  return true;
}

bool tarakanov_d_global_opt_two_dim_prob_seq::GlobalOptSequential::PostProcessingImpl() {
  *reinterpret_cast<double*>(task_data->outputs[0]) = result;
  task_data->outputs_count[0] = 1;
  return true;
}