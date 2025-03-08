// @copyright Tarakanov Denis
#include "seq/tarakanov_d_global_opt_by_characts_two_dim_prob/include/ops_seq.hpp"

#include <limits>
#include <vector>

double tarakanov_d_global_opt_two_dim_prob_seq::GetConstraintsSum(double x, double y, int num,
                                                                  std::vector<double> vec) {
  return vec[num * 3] * x + vec[(num * 3) + 1] * y - vec[(num * 3) + 2];
}

bool tarakanov_d_global_opt_two_dim_prob_seq::CheckConstraints(double x, double y, int constraint_num,
                                                               std::vector<double> constr) {
  return GetConstraintsSum(x, y, constraint_num, constr) <= 0;
}

bool tarakanov_d_global_opt_two_dim_prob_seq::IsAcceptable(double x, double y, int constraint_num,
                                                           std::vector<double> constr) {
  for (int i = 0; i < constraint_num; ++i) {
    if (false == CheckConstraints(x, y, i, constr)) {
      return false;
    }
  }

  return true;
}

double tarakanov_d_global_opt_two_dim_prob_seq::ComputeFunction(double x, double y, std::vector<double> params) {
  return ((x - params[0]) * (x - params[0])) + ((y - params[1]) * (y - params[1]));
}

bool tarakanov_d_global_opt_two_dim_prob_seq::GlobalOptSequential::ValidationImpl() {
  bool rc = (1 == task_data->outputs_count[0]);
  rc = rc && (2 == task_data->inputs_count.size());
  return rc;
}

bool tarakanov_d_global_opt_two_dim_prob_seq::GlobalOptSequential::PreProcessingImpl() {
  bounds_.push_back(reinterpret_cast<double*>(task_data->inputs[0])[0]);
  bounds_.push_back(reinterpret_cast<double*>(task_data->inputs[0])[1]);
  bounds_.push_back(reinterpret_cast<double*>(task_data->inputs[0])[2]);
  bounds_.push_back(reinterpret_cast<double*>(task_data->inputs[0])[3]);

  constr_num_ = task_data->inputs_count[0];
  constr_.resize(constr_num_ * 3);
  mode_ = task_data->inputs_count[1];

  params_.push_back(reinterpret_cast<double*>(task_data->inputs[1])[0]);
  params_.push_back(reinterpret_cast<double*>(task_data->inputs[1])[1]);

  for (int i = 0; i < constr_num_ * 3; i++) {
    constr_[i] = (reinterpret_cast<double*>(task_data->inputs[2])[i]);
  }

  delta_ = *reinterpret_cast<double*>(task_data->inputs[3]);

  return true;
}

bool tarakanov_d_global_opt_two_dim_prob_seq::GlobalOptSequential::RunImpl() {
  switch (mode_) {
    case 1:
      result_ = std::numeric_limits<double>::min();
      break;
    case 0:
      result_ = std::numeric_limits<double>::max();
      break;
  }
  double accuracy = 1e-6;
  double last_result = std::numeric_limits<double>::max();
  auto factor = static_cast<int>(1.0 / delta_);

  for (; delta_ >= accuracy; delta_ /= 2.0) {
    double local_minX = bounds_[0];
    double local_minY = bounds_[2];

    auto int_minX = static_cast<int>(bounds_[0] * factor);
    auto x = int_minX;
    auto int_maxX = static_cast<int>(bounds_[1] * factor);

    auto int_minY = static_cast<int>(bounds_[2] * factor);
    auto y = int_minY;
    auto int_maxY = static_cast<int>(bounds_[3] * factor);

    while (x < int_maxX) {
      auto real_x = static_cast<double>(++x) / factor;
      while (y < int_maxY) {
        auto real_y = static_cast<double>(++y) / factor;

        if (true == IsAcceptable(real_x, real_y, constr_num_, constr_)) {
          auto value = ComputeFunction(real_x, real_y, params_);
          SaveResult(mode_, value, local_minX, local_minY, real_x, real_y);
        }
      }
    }

    if (std::abs(last_result - result_) < accuracy) {
      break;
    }

    bounds_[0] = std::max(local_minX - 2 * delta_, bounds_[0]);
    bounds_[1] = std::min(local_minX + 2 * delta_, bounds_[1]);
    bounds_[2] = std::max(local_minY - 2 * delta_, bounds_[2]);
    bounds_[3] = std::min(local_minY + 2 * delta_, bounds_[3]);

    last_result = result_;
  }

  return true;
}

void tarakanov_d_global_opt_two_dim_prob_seq::GlobalOptSequential::SaveResult(int mode, double value, double& local_min_x, double& local_min_y, double real_x, double real_y){
  switch (mode) {
    case 0:
      if (value < result_) {
        result_ = value;
        local_min_x = real_x;
        local_min_y = real_y;
      }
      break;
    case 1:
      if (value > result_) {
        result_ = value;
        local_min_x = real_x;
        local_min_y = real_y;
      }
      break;
  }
}

bool tarakanov_d_global_opt_two_dim_prob_seq::GlobalOptSequential::PostProcessingImpl() {
  *reinterpret_cast<double*>(task_data->outputs[0]) = result_;
  task_data->outputs_count[0] = 1;
  return true;
}