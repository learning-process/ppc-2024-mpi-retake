#include "mpi/prokhorov_n_global_search_algorithm_strongin/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/mpi.hpp>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <thread>
#include <vector>

namespace prokhorov_n_global_search_algorithm_strongin_mpi {

bool prokhorov_n_global_search_algorithm_strongin_mpi::TestTaskSequential::PreProcessingImpl() {
  a = reinterpret_cast<double*>(task_data->inputs[0])[0];
  b = reinterpret_cast<double*>(task_data->inputs[1])[0];
  epsilon = reinterpret_cast<double*>(task_data->inputs[2])[0];

  f = [](double x) { return x * x; };

  result = 0.0;
  return true;
}

bool prokhorov_n_global_search_algorithm_strongin_mpi::TestTaskSequential::ValidationImpl() {
  return task_data->inputs_count[0] == 1 && task_data->inputs_count[1] == 1 && task_data->inputs_count[2] == 1 &&
         task_data->outputs_count[0] == 1;
}

bool prokhorov_n_global_search_algorithm_strongin_mpi::TestTaskSequential::RunImpl() {
  result = stronginAlgorithm();
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  return true;
}

bool prokhorov_n_global_search_algorithm_strongin_mpi::TestTaskSequential::PostProcessingImpl() {
  reinterpret_cast<double*>(task_data->outputs[0])[0] = result;
  return true;
}

double prokhorov_n_global_search_algorithm_strongin_mpi::TestTaskSequential::stronginAlgorithm() {
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

bool prokhorov_n_global_search_algorithm_strongin_mpi::TestTaskMPI::PreProcessingImpl() {
  if (world.rank() == 0) {
    result = 0;
    a = *reinterpret_cast<double*>(task_data->inputs[0]);
    b = *reinterpret_cast<double*>(task_data->inputs[1]);
    epsilon = *reinterpret_cast<double*>(task_data->inputs[2]);
  }

  f = [](double x) { return x * x; };

  return true;
}

bool prokhorov_n_global_search_algorithm_strongin_mpi::TestTaskMPI::ValidationImpl() {
  if (world.rank() == 0) {
    return task_data->outputs_count[0] == 1;
  }
  return true;
}

bool prokhorov_n_global_search_algorithm_strongin_mpi::TestTaskMPI::RunImpl() {
  if (world.size() == 1) {
    result = stronginAlgorithm();
  } else {
    result = stronginAlgorithmParallel();
  }

  return true;
}
double prokhorov_n_global_search_algorithm_strongin_mpi::TestTaskMPI::stronginAlgorithmParallel() {
  double local_a, local_b;
  int rank = world.rank();
  int size = world.size();

  double interval_length = (b - a) / size;
  local_a = a + rank * interval_length;
  local_b = local_a + interval_length;

  double local_x_min = local_a;
  double local_f_min = f(local_x_min);

  while ((local_b - local_a) > epsilon) {
    double x1 = local_a + (local_b - local_a) / 3.0;
    double x2 = local_b - (local_b - local_a) / 3.0;

    double f1 = f(x1);
    double f2 = f(x2);

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
  boost::mpi::gather(world, local_x_min, all_x_mins, 0);

  if (rank == 0) {
    double global_x_min = all_x_mins[0];
    double global_f_min = f(global_x_min);

    for (int i = 1; i < size; ++i) {
      double current_f = f(all_x_mins[i]);
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
  if (world.rank() == 0) {
    reinterpret_cast<double*>(task_data->outputs[0])[0] = result;
  }
  return true;
}
double prokhorov_n_global_search_algorithm_strongin_mpi::TestTaskMPI::stronginAlgorithm() {
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
}  // namespace prokhorov_n_global_search_algorithm_strongin_mpi