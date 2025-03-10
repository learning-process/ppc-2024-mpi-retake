#include "mpi/tarakanov_d_global_opt_by_characts_two_dim_prob/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/mpi/collectives/broadcast.hpp>
#include <boost/mpi/collectives/gather.hpp>
#include <boost/mpi/communicator.hpp>
#include <cmath>
#include <limits>
#include <vector>

double tarakanov_d_global_opt_two_dim_prob_mpi::GetConstraintsSum(double x, double y, int num,
                                                                  std::vector<double> vec) {
  return (vec[num * 3] * x) + (vec[(num * 3) + 1] * y) - vec[(num * 3) + 2];
}

bool tarakanov_d_global_opt_two_dim_prob_mpi::CheckConstraints(double x, double y, int constraint_num,
                                                               const std::vector<double>& constraints) {
  return GetConstraintsSum(x, y, constraint_num, constraints) <= 0;
}

bool tarakanov_d_global_opt_two_dim_prob_mpi::IsAcceptable(double x, double y, int constraint_num,
                                                           const std::vector<double>& constraints) {
  for (int i = 0; i < constraint_num; ++i) {
    if (!CheckConstraints(x, y, i, constraints)) {
      return false;
    }
  }
  return true;
}

double tarakanov_d_global_opt_two_dim_prob_mpi::ComputeFunction(double x, double y, std::vector<double> params) {
  return ((x - params[0]) * (x - params[0])) + ((y - params[1]) * (y - params[1]));
}

bool tarakanov_d_global_opt_two_dim_prob_mpi::GlobalOptSequential::ValidationImpl() {
  return (1 == task_data->outputs_count[0]) && (2 == task_data->inputs_count.size());
}

bool tarakanov_d_global_opt_two_dim_prob_mpi::GlobalOptSequential::PreProcessingImpl() {
  bounds_.push_back(reinterpret_cast<double*>(task_data->inputs[0])[0]);
  bounds_.push_back(reinterpret_cast<double*>(task_data->inputs[0])[1]);
  bounds_.push_back(reinterpret_cast<double*>(task_data->inputs[0])[2]);
  bounds_.push_back(reinterpret_cast<double*>(task_data->inputs[0])[3]);

  constr_num_ = static_cast<int>(task_data->inputs_count[0]);
  constr_.resize(constr_num_ * 3);
  mode_ = static_cast<int>(task_data->inputs_count[1]);

  params_.push_back(reinterpret_cast<double*>(task_data->inputs[1])[0]);
  params_.push_back(reinterpret_cast<double*>(task_data->inputs[1])[1]);

  for (int i = 0; i < constr_num_ * 3; i++) {
    constr_[i] = reinterpret_cast<double*>(task_data->inputs[2])[i];
  }

  delta_ = *reinterpret_cast<double*>(task_data->inputs[3]);

  return true;
}

bool tarakanov_d_global_opt_two_dim_prob_mpi::GlobalOptSequential::RunImpl() {
  switch (mode_) {
    case 1:
      result_ = std::numeric_limits<double>::min();
      break;
    case 0:
      result_ = std::numeric_limits<double>::max();
      break;
    default:
      return false;
      break;
  }

  double current_step = delta_;
  double accuracy = 1e-6;
  double last_result = std::numeric_limits<double>::max();
  int factor = static_cast<int>(1.0 / current_step);

  while (current_step >= accuracy) {
    double local_min_x = bounds_[0];
    double local_min_y = bounds_[2];

    int int_min_x = static_cast<int>(bounds_[0] * factor);
    int int_max_x = static_cast<int>(bounds_[1] * factor);
    int int_min_y = static_cast<int>(bounds_[2] * factor);
    int int_max_y = static_cast<int>(bounds_[3] * factor);

    ProcessGridPoints(int_min_x, int_max_x, int_min_y, int_max_y, factor, local_min_x, local_min_y);

    if (std::abs(last_result - result_) < accuracy) {
      break;
    }

    bounds_[0] = std::max(local_min_x - (2 * current_step), bounds_[0]);
    bounds_[1] = std::min(local_min_x + (2 * current_step), bounds_[1]);
    bounds_[2] = std::max(local_min_y - (2 * current_step), bounds_[2]);
    bounds_[3] = std::min(local_min_y + (2 * current_step), bounds_[3]);

    current_step /= 2.0;
    last_result = result_;
  }

  return true;
}

void tarakanov_d_global_opt_two_dim_prob_mpi::GlobalOptSequential::ProcessGridPoints(int int_min_x, int int_max_x,
                                                                                     int int_min_y, int int_max_y,
                                                                                     int factor, double& local_min_x,
                                                                                     double& local_min_y) {
  int x = int_min_x;
  while (x < int_max_x) {
    int y = int_min_y;
    while (y < int_max_y) {
      double real_x = x++ / static_cast<double>(factor);
      double real_y = y++ / static_cast<double>(factor);

      if (IsPointCorrect(real_x, real_y)) {
        double value = ComputeFunction(real_x, real_y, params_);
        if (mode_ == 0) {
          if (value < result_) {
            result_ = value;
            local_min_x = real_x;
            local_min_y = real_y;
          }
        } else if (mode_ == 1) {
          if (value > result_) {
            result_ = value;
            local_min_x = real_x;
            local_min_y = real_y;
          }
        }
      }
    }
  }
}

bool tarakanov_d_global_opt_two_dim_prob_mpi::GlobalOptSequential::IsPointCorrect(double real_x, double real_y) {
  for (int i = 0; i < constr_num_; i++) {
    if (!CheckConstraints(real_x, real_y, i, constr_)) {
      return false;
    }
  }
  return true;
}

bool tarakanov_d_global_opt_two_dim_prob_mpi::GlobalOptSequential::PostProcessingImpl() {
  *reinterpret_cast<double*>(task_data->outputs[0]) = result_;
  task_data->outputs_count[0] = 1;
  return true;
}

bool tarakanov_d_global_opt_two_dim_prob_mpi::GlobalOptMpi::ValidationImpl() {
  if (world_.rank() == 0) {
    if (task_data->outputs_count[0] == 0) {
      return false;
    }
    if ((task_data->inputs_count[1] != 1) && (task_data->inputs_count[1] != 0)) {
      return false;
    }
  }
  return true;
}

bool tarakanov_d_global_opt_two_dim_prob_mpi::GlobalOptMpi::PreProcessingImpl() {
  if (world_.rank() == 0) {
    delta_ = *reinterpret_cast<double*>(task_data->inputs[3]);
    bounds_.resize(4);
    for (int i = 0; i < 4; i++) {
      bounds_[i] = reinterpret_cast<double*>(task_data->inputs[0])[i];
    }

    constr_num_ = static_cast<int>(task_data->inputs_count[0]);
    mode_ = static_cast<int>(task_data->inputs_count[1]);

    for (int i = 0; i < 2; i++) {
      params_.push_back(reinterpret_cast<double*>(task_data->inputs[1])[i]);
    }
    for (int i = 0; i < constr_num_ * 3; i++) {
      constr_.push_back(reinterpret_cast<double*>(task_data->inputs[2])[i]);
    }
  }
  local_constr_size_ = 0;
  return true;
}

bool tarakanov_d_global_opt_two_dim_prob_mpi::GlobalOptMpi::RunImpl() {
  if (world_.rank() == 0) {
    switch (mode_) {
      case 1:
        result_ = std::numeric_limits<double>::min();
        break;
      case 0:
        result_ = std::numeric_limits<double>::max();
        break;
      default:
        return false;
        break;
    }
  }
  broadcast(world_, result_, 0);

  if (world_.rank() == 0) {
    local_constr_size_ = std::max(1, (constr_num_ + world_.size() - 1) / world_.size());
  }

  broadcast(world_, constr_num_, 0);
  broadcast(world_, local_constr_size_, 0);
  broadcast(world_, delta_, 0);

  if (world_.rank() != 0) {
    bounds_.resize(4);
  }
  broadcast(world_, bounds_.data(), static_cast<int>(bounds_.size()), 0);

  DataDistribution();

  double current_step = delta_;
  double accuracy = 1e-6;
  double last_result = std::numeric_limits<double>::max();
  std::vector<double> loc_area(4, 0);
  for (int i = 0; i < 4; i++) {
    loc_area[i] = bounds_[i];
  }

  while (current_step >= accuracy) {
    double local_min_x = loc_area[0];
    double local_min_y = loc_area[2];

    int factor = static_cast<int>(1.0 / current_step);
    int int_min_x = static_cast<int>(loc_area[0] * factor);
    int int_max_x = static_cast<int>(loc_area[1] * factor);
    int int_min_y = static_cast<int>(loc_area[2] * factor);
    int int_max_y = static_cast<int>(loc_area[3] * factor);

    ProccessGridPoint(int_min_x, int_min_y, int_max_x, int_max_y, factor, local_min_x, local_min_y);

    if (world_.rank() == 0) {
      NewAreaProcess(last_result, loc_area, local_min_x, local_min_y, current_step, accuracy);
    }

    broadcast(world_, loc_area.data(), static_cast<int>(bounds_.size()), 0);
    broadcast(world_, current_step, 0);
  }

  return true;
}

void tarakanov_d_global_opt_two_dim_prob_mpi::GlobalOptMpi::NewAreaProcess(double& last_result,
                                                                           std::vector<double>& loc_area,
                                                                           double local_min_x, double local_min_y,
                                                                           double& current_step, double accuracy) {
  if (std::abs(last_result - result_) < accuracy) {
    current_step = -1;
  }
  std::vector<double> new_area = loc_area;
  new_area[0] = std::max(local_min_x - (2 * current_step), bounds_[0]);
  new_area[1] = std::min(local_min_x + (2 * current_step), bounds_[1]);
  new_area[2] = std::max(local_min_y - (2 * current_step), bounds_[2]);
  new_area[3] = std::min(local_min_y + (2 * current_step), bounds_[3]);
  loc_area = new_area;
  last_result = result_;
}

void tarakanov_d_global_opt_two_dim_prob_mpi::GlobalOptMpi::DataDistribution() {
  if (world_.rank() == 0) {
    for (int pr = 1; pr < world_.size(); pr++) {
      if (pr * local_constr_size_ < constr_num_) {
        std::vector<double> send(3 * local_constr_size_, 0);
        for (int i = 0; i < 3 * local_constr_size_; i++) {
          send[i] = constr_[(pr * local_constr_size_ * 3) + i];
        }
        world_.send(pr, 0, send.data(), static_cast<int>(send.size()));
      }
    }
    for (int i = 0; i < 3 * local_constr_size_; i++) {
      local_constr_.push_back(constr_[i]);
    }
  } else if (world_.rank() < constr_num_) {
    std::vector<double> buffer(local_constr_size_ * 3, 0);
    world_.recv(0, 0, buffer.data(), static_cast<int>(buffer.size()));
    local_constr_.insert(local_constr_.end(), buffer.begin(), buffer.end());
  }
}

int tarakanov_d_global_opt_two_dim_prob_mpi::GlobalOptMpi::ApproveAllConstraints(double real_x, double real_y,
                                                                                 int constr_sz) {
  for (int i = 0; i < constr_sz; i++) {
    if (!CheckConstraints(real_x, real_y, i, local_constr_)) {
      return 0;
    }
  }
  return 1;
}

void tarakanov_d_global_opt_two_dim_prob_mpi::GlobalOptMpi::ProccessGridPoint(int int_min_x, int int_min_y,
                                                                              int int_max_x, int int_max_y, int factor,
                                                                              double& local_min_x,
                                                                              double& local_min_y) {
  int x = int_min_x;
  while (x < int_max_x) {
    int y = int_min_y;
    while (y < int_max_y) {
      double real_x = x / static_cast<double>(factor);
      double real_y = y / static_cast<double>(factor);

      int constr_sz = static_cast<int>(local_constr_.size()) / 3;
      int loc_flag = ApproveAllConstraints(real_x, real_y, constr_sz);

      gather(world_, loc_flag, is_correct_, 0);

      if (world_.rank() == 0) {
        bool flag = true;
        int sz = static_cast<int>(is_correct_.size());
        flag = CheckCorrect(sz);

        if (flag) {
          double value = ComputeFunction(real_x, real_y, params_);
          SaveResult(real_x, real_y, value, local_min_x, local_min_y);
        }
      }
      y++;
    }
    x++;
  }
}

void tarakanov_d_global_opt_two_dim_prob_mpi::GlobalOptMpi::SaveResult(double real_x, double real_y, double value,
                                                                       double& local_min_x, double& local_min_y) {
  if (mode_ == 0) {
    if (value < result_) {
      result_ = value;
      local_min_x = real_x;
      local_min_y = real_y;
    }
  } else if (mode_ == 1) {
    if (value > result_) {
      result_ = value;
      local_min_x = real_x;
      local_min_y = real_y;
    }
  }
}

bool tarakanov_d_global_opt_two_dim_prob_mpi::GlobalOptMpi::CheckCorrect(int sz) {
  for (int i = 0; i < sz; i++) {
    if (is_correct_[i] == 0) {
      return false;
    }
  }
  return true;
}

bool tarakanov_d_global_opt_two_dim_prob_mpi::GlobalOptMpi::PostProcessingImpl() {
  if (world_.rank() == 0) {
    reinterpret_cast<double*>(task_data->outputs[0])[0] = result_;
  }
  return true;
}