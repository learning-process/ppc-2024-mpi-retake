#include "mpi/tarakanov_d_global_opt_by_characts_two_dim_prob/include/ops_mpi.hpp"
#include <limits>

double tarakanov_d_global_opt_two_dim_prob_mpi::GetConstraintsSum(double x, double y, int num, std::vector<double> vec) {
    return vec[num * 3] * x + vec[num * 3 + 1] * y - vec[num * 3 + 2];
}

bool tarakanov_d_global_opt_two_dim_prob_mpi::CheckConstraints(double x, double y, int constraint_num,
                                                                std::vector<double> constraints) {
    return GetConstraintsSum(x, y, constraint_num, constraints) <= 0;
}

bool tarakanov_d_global_opt_two_dim_prob_mpi::IsAcceptable(double x, double y, int constraint_num, std::vector<double> constraints) {
    for (int i = 0; i < constraint_num; ++i) {
        if (!CheckConstraints(x, y, i, constraints)) {
            return false;
        }
    }
    return true;
}

double tarakanov_d_global_opt_two_dim_prob_mpi::ComputeFunction(double x, double y, std::vector<double> params) {
    return (x - params[0]) * (x - params[0]) + (y - params[1]) * (y - params[1]);
}

bool tarakanov_d_global_opt_two_dim_prob_mpi::GlobalOptSequential::ValidationImpl() {
    return (1 == task_data->outputs_count[0]) &&
           (2 == task_data->inputs_count.size());
}

bool tarakanov_d_global_opt_two_dim_prob_mpi::GlobalOptSequential::PreProcessingImpl() {
    bounds.push_back(reinterpret_cast<double*>(task_data->inputs[0])[0]);
    bounds.push_back(reinterpret_cast<double*>(task_data->inputs[0])[1]);
    bounds.push_back(reinterpret_cast<double*>(task_data->inputs[0])[2]);
    bounds.push_back(reinterpret_cast<double*>(task_data->inputs[0])[3]);

    constr_num = task_data->inputs_count[0];
    constr.resize(constr_num * 3);
    mode = task_data->inputs_count[1];

    params.push_back(reinterpret_cast<double*>(task_data->inputs[1])[0]);
    params.push_back(reinterpret_cast<double*>(task_data->inputs[1])[1]);

    for (int i = 0; i < constr_num * 3; i++) {
        constr[i] = reinterpret_cast<double*>(task_data->inputs[2])[i];
    }

    delta = *reinterpret_cast<double*>(task_data->inputs[3]);

    return true;
}


bool tarakanov_d_global_opt_two_dim_prob_mpi::GlobalOptSequential::RunImpl() {
  switch (mode) {
      case 1:
          result = std::numeric_limits<double>::min();
          break;
      case 0:
          result = std::numeric_limits<double>::max();
          break;
  }

  double current_step = delta;
  double accuracy = 1e-6;
  double last_result = std::numeric_limits<double>::max();
  int factor = static_cast<int>(1.0 / current_step);

  while (current_step >= accuracy) {
      double local_minX = bounds[0];
      double local_minY = bounds[2];

      int int_minX = static_cast<int>(bounds[0] * factor);
      int x = int_minX;
      int int_maxX = static_cast<int>(bounds[1] * factor);

      int int_minY = static_cast<int>(bounds[2] * factor);
      int int_maxY = static_cast<int>(bounds[3] * factor);

      while (x < int_maxX) {
          int y = int_minY;
          while (y < int_maxY) {
              double real_x = x++ / static_cast<double>(factor);
              double real_y = y++ / static_cast<double>(factor);
              bool is_point_correct = true;

              for (int i = 0; i < constr_num; i++) {
                  if (!CheckConstraints(real_x, real_y, i, constr)) {
                      is_point_correct = false;
                      break;
                  }
              }

              if (is_point_correct) {
                  double value = ComputeFunction(real_x, real_y, params);
                  if (mode == 0) {
                      if (value < result) {
                          result = value;
                          local_minX = real_x;
                          local_minY = real_y;
                      }
                  } else if (mode == 1) {
                      if (value > result) {
                          result = value;
                          local_minX = real_x;
                          local_minY = real_y;
                      }
                  }
              }
          }
      }

      if (std::abs(last_result - result) < accuracy) {
          break;
      }

      bounds[0] = std::max(local_minX - 2 * current_step, bounds[0]);
      bounds[1] = std::min(local_minX + 2 * current_step, bounds[1]);
      bounds[2] = std::max(local_minY - 2 * current_step, bounds[2]);
      bounds[3] = std::min(local_minY + 2 * current_step, bounds[3]);

      current_step /= 2.0;
      last_result = result;
  }

  return true;
}

bool tarakanov_d_global_opt_two_dim_prob_mpi::GlobalOptSequential::PostProcessingImpl() {
    *reinterpret_cast<double*>(task_data->outputs[0]) = result;
    task_data->outputs_count[0] = 1;
    return true;
}

bool tarakanov_d_global_opt_two_dim_prob_mpi::GlobalOptMpi::ValidationImpl() {
    if (world.rank() == 0) {
        if (task_data->outputs_count[0] == 0)
            return false;
        if ((task_data->inputs_count[1] != 1) && (task_data->inputs_count[1] != 0))
            return false;
    }
    return true;
}

bool tarakanov_d_global_opt_two_dim_prob_mpi::GlobalOptMpi::PreProcessingImpl() {
    if (world.rank() == 0) {
        delta = *reinterpret_cast<double*>(task_data->inputs[3]);
        bounds.resize(4);
        for (int i = 0; i < 4; i++) {
            bounds[i] = reinterpret_cast<double*>(task_data->inputs[0])[i];
        }

        constr_num = task_data->inputs_count[0];
        mode = task_data->inputs_count[1];

        for (int i = 0; i < 2; i++) {
            params.push_back(reinterpret_cast<double*>(task_data->inputs[1])[i]);
        }
        for (int i = 0; i < constr_num * 3; i++) {
            constr.push_back(reinterpret_cast<double*>(task_data->inputs[2])[i]);
        }
    }
    local_constr_size = 0;
    return true;
}

bool tarakanov_d_global_opt_two_dim_prob_mpi::GlobalOptMpi::RunImpl() {
  switch (mode) {
      case 1:
          result = std::numeric_limits<double>::min();
          break;
      case 0:
          result = std::numeric_limits<double>::max();
          break;
  }

  if (world.rank() == 0) {
      local_constr_size = std::max(1, (constr_num + world.size() - 1) / world.size());
  }

  broadcast(world, constr_num, 0);
  broadcast(world, local_constr_size, 0);
  broadcast(world, delta, 0);

  if (world.rank() != 0) {
      bounds.resize(4);
  }
  broadcast(world, bounds.data(), bounds.size(), 0);

  if (world.rank() == 0) {
      for (int pr = 1; pr < world.size(); pr++) {
          if (pr * local_constr_size < constr_num) {
              std::vector<double> send(3 * local_constr_size, 0);
              for (int i = 0; i < 3 * local_constr_size; i++) {
                  send[i] = constr[pr * local_constr_size * 3 + i];
              }
              world.send(pr, 0, send.data(), send.size());
          }
      }
      for (int i = 0; i < 3 * local_constr_size; i++) {
          local_constr.push_back(constr[i]);
      }
  } else if (world.rank() < constr_num) {
      std::vector<double> buffer(local_constr_size * 3, 0);
      world.recv(0, 0, buffer.data(), buffer.size());
      local_constr.insert(local_constr.end(), buffer.begin(), buffer.end());
  }

  double current_step = delta;
  double accuracy = 1e-6;
  double last_result = std::numeric_limits<double>::max();
  std::vector<double> loc_area(4, 0);
  for (int i = 0; i < 4; i++) {
      loc_area[i] = bounds[i];
  }

  while (current_step >= accuracy) {
      double local_minX = loc_area[0];
      double local_minY = loc_area[2];

      int factor = static_cast<int>(1.0 / current_step);
      int int_minX = static_cast<int>(loc_area[0] * factor);
      int int_maxX = static_cast<int>(loc_area[1] * factor);
      int int_minY = static_cast<int>(loc_area[2] * factor);
      int int_maxY = static_cast<int>(loc_area[3] * factor);

      int x = int_minX;
      while (x < int_maxX) {
          int y = int_minY;
          while (y < int_maxY) {
              double real_x = x / static_cast<double>(factor);
              double real_y = y / static_cast<double>(factor);

              int loc_flag = 1;
              int constr_sz = local_constr.size() / 3;
              for (int i = 0; i < constr_sz; i++) {
                  if (!CheckConstraints(real_x, real_y, i, local_constr)) {
                      loc_flag = 0;
                      break;
                  }
              }

              gather(world, loc_flag, is_corret, 0);

              if (world.rank() == 0) {
                  bool flag = true;
                  int sz = is_corret.size();
                  for (int i = 0; i < sz; i++) {
                      if (is_corret[i] == 0) {
                          flag = false;
                          break;
                      }
                  }
                  if (flag) {
                      double value = ComputeFunction(real_x, real_y, params);
                      if (mode == 0) {
                          if (value < result) {
                              result = value;
                              local_minX = real_x;
                              local_minY = real_y;
                          }
                      } else if (mode == 1) {
                          if (value > result) {
                              result = value;
                              local_minX = real_x;
                              local_minY = real_y;
                          }
                      }
                  }
              }
              y++;
          }
          x++;
      }

      if (world.rank() == 0) {
          if (std::abs(last_result - result) < accuracy) {
              current_step = -1;
          }
          std::vector<double> new_area = loc_area;
          new_area[0] = std::max(local_minX - 2 * current_step, bounds[0]);
          new_area[1] = std::min(local_minX + 2 * current_step, bounds[1]);
          new_area[2] = std::max(local_minY - 2 * current_step, bounds[2]);
          new_area[3] = std::min(local_minY + 2 * current_step, bounds[3]);
          loc_area = new_area;
          last_result = result;
      }

      broadcast(world, loc_area.data(), bounds.size(), 0);
      broadcast(world, current_step, 0);
  }

  return true;
}

bool tarakanov_d_global_opt_two_dim_prob_mpi::GlobalOptMpi::PostProcessingImpl() {
    if (world.rank() == 0) {
        reinterpret_cast<double*>(task_data->outputs[0])[0] = result;
    }
    return true;
}