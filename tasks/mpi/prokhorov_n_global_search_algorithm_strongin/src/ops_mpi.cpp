#include "mpi/prokhorov_n_global_search_algorithm_strongin/include/ops_mpi.hpp"

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/collectives/gather.hpp>
#include <boost/mpi/communicator.hpp>
#include <chrono>
#include <cmath>
#include <functional>
#include <thread>
#include <vector>

namespace prokhorov_n_global_search_algorithm_strongin_mpi {

bool prokhorov_n_global_search_algorithm_strongin_mpi::TestTaskSequential::PreProcessingImpl() {
  a_ = reinterpret_cast<double*>(task_data->inputs[0])[0];
  b_ = reinterpret_cast<double*>(task_data->inputs[1])[0];
  epsilon_ = reinterpret_cast<double*>(task_data->inputs[2])[0];

  f_ = [](double x) { return x * x; };

  result_ = 0.0;
  return true;
}

bool prokhorov_n_global_search_algorithm_strongin_mpi::TestTaskSequential::ValidationImpl() {
  return task_data->inputs_count[0] == 1 && task_data->inputs_count[1] == 1 && task_data->inputs_count[2] == 1 &&
         task_data->outputs_count[0] == 1;
}

bool prokhorov_n_global_search_algorithm_strongin_mpi::TestTaskSequential::RunImpl() {
  result_ = StronginAlgorithm();
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  return true;
}

bool prokhorov_n_global_search_algorithm_strongin_mpi::TestTaskSequential::PostProcessingImpl() {
  reinterpret_cast<double*>(task_data->outputs[0])[0] = result_;
  return true;
}

double prokhorov_n_global_search_algorithm_strongin_mpi::TestTaskSequential::StronginAlgorithm() {
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

bool prokhorov_n_global_search_algorithm_strongin_mpi::TestTaskMPI::PreProcessingImpl() {
  if (world_.rank() == 0) {
    result_ = 0;
    a_ = *reinterpret_cast<double*>(task_data->inputs[0]);
    b_ = *reinterpret_cast<double*>(task_data->inputs[1]);
    epsilon_ = *reinterpret_cast<double*>(task_data->inputs[2]);
  }

  f_ = [](double x) { return x * x; };

  return true;
}

bool prokhorov_n_global_search_algorithm_strongin_mpi::TestTaskMPI::ValidationImpl() {
  if (world_.rank() == 0) {
    return task_data->outputs_count[0] == 1;
  }
  return true;
}

bool prokhorov_n_global_search_algorithm_strongin_mpi::TestTaskMPI::RunImpl() {
  if (world_.size() == 1) {
    result_ = StronginAlgorithm();
  } else {
    result_ = StronginAlgorithmParallel();
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  return true;
}

double prokhorov_n_global_search_algorithm_strongin_mpi::TestTaskMPI::StronginAlgorithmParallel() {
  double local_a = 0.0;
  double local_b = 0.0;
  int rank = world_.rank();
  int size = world_.size();

  double interval_length = (b_ - a_) / size;
  local_a = a_ + rank * interval_length;
  local_b = local_a + interval_length;

  double local_x_min = local_a;
  double local_f_min = f_(local_x_min);

  while ((local_b - local_a) > epsilon_) {
    double x1 = local_a + ((local_b - local_a) / 3.0);
    double x2 = local_b - ((local_b - local_a) / 3.0);

    double f1 = f_(x1);
    double f2 = f_(x2);

    if (f1 < f2) {
      local_b = x2;
      if (f1 < local_f_min) {
        local_f_min = f1;
        local_x_min = x1;
      }
    } else {
      local_a = x1;
      if (f2 < local_f_min) {
        local_f_min = f2;
        local_x_min = x2;
      }
    }
  }

  std::vector<double> all_x_mins(size);
  boost::mpi::gather(world_, local_x_min, all_x_mins, 0);

  if (rank == 0) {
    double global_x_min = all_x_mins[0];
    double global_f_min = f_(global_x_min);

    for (int i = 1; i < size; ++i) {
      double current_f = f_(all_x_mins[i]);
      if (current_f < global_f_min) {
        global_f_min = current_f;
        global_x_min = all_x_mins[i];
      }
    }

    return global_x_min;
  }

  return 0.0;
}

bool prokhorov_n_global_search_algorithm_strongin_mpi::TestTaskMPI::PostProcessingImpl() {
  if (world_.rank() == 0) {
    reinterpret_cast<double*>(task_data->outputs[0])[0] = result_;
  }
  return true;
}

double prokhorov_n_global_search_algorithm_strongin_mpi::TestTaskMPI::StronginAlgorithm() {
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

}  // namespace prokhorov_n_global_search_algorithm_strongin_mpi