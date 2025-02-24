#include "mpi/prokhorov_n_global_search_algorithm_strongin/include/ops_mpi.hpp"

#include <algorithm>
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
    if (task_data->inputs.empty() || task_data->inputs.size() < 3) {
      throw std::runtime_error("Not enough input data.");
    }

    if (task_data->inputs[0] == nullptr || task_data->inputs[1] == nullptr || task_data->inputs[2] == nullptr) {
      throw std::runtime_error("Input data is null.");
    }

    a = *reinterpret_cast<double*>(task_data->inputs[0]);
    b = *reinterpret_cast<double*>(task_data->inputs[1]);
    epsilon = *reinterpret_cast<double*>(task_data->inputs[2]);
  }

  boost::mpi::broadcast(world, a, 0);
  boost::mpi::broadcast(world, b, 0);
  boost::mpi::broadcast(world, epsilon, 0);

  if (world.rank() != 0) {
    if (a == 0.0 && b == 0.0 && epsilon == 0.0) {
      throw std::runtime_error("Data was not broadcasted correctly.");
    }
  }

  f = [](double x) { return x * x; };

  result = 0.0;
  return true;
}

bool prokhorov_n_global_search_algorithm_strongin_mpi::TestTaskMPI::ValidationImpl() {
  return task_data->inputs_count[0] == 1 && task_data->inputs_count[1] == 1 && task_data->inputs_count[2] == 1 &&
         task_data->outputs_count[0] == 1;
}

bool prokhorov_n_global_search_algorithm_strongin_mpi::TestTaskMPI::RunImpl() {
  if (world.size() == 1) {
    result = stronginAlgorithm();
  } else {
    result = stronginAlgorithmParallel();
  }

  return true;
}

bool prokhorov_n_global_search_algorithm_strongin_mpi::TestTaskMPI::PostProcessingImpl() {
  if (world.rank() == 0) {
    reinterpret_cast<double*>(task_data->outputs[0])[0] = result;
  }
  return true;
}

double prokhorov_n_global_search_algorithm_strongin_mpi::TestTaskMPI::stronginAlgorithmParallel() {
  double global_min = std::numeric_limits<double>::max();
  double global_x_min = a;

  while ((b - a) > epsilon) {
    double step = (b - a) / world.size();
    double local_a = a + step * world.rank();
    double local_b = a + step * (world.rank() + 1);

    double x1 = local_a + (local_b - local_a) / 3.0;
    double x2 = local_b - (local_b - local_a) / 3.0;

    double f1 = f(x1);
    double f2 = f(x2);

    double local_min, local_x_min;
    if (f1 < f2) {
      local_min = f1;
      local_x_min = x1;
    } else {
      local_min = f2;
      local_x_min = x2;
    }

    std::vector<double> all_mins(world.size());
    std::vector<double> all_x_mins(world.size());
    boost::mpi::gather(world, local_min, all_mins, 0);
    boost::mpi::gather(world, local_x_min, all_x_mins, 0);

    world.barrier();

    if (world.rank() == 0) {
      for (int i = 0; i < world.size(); ++i) {
        if (all_mins[i] < global_min) {
          global_min = all_mins[i];
          global_x_min = all_x_mins[i];
        }
      }

      a = global_x_min - step;
      b = global_x_min + step;
    }

    boost::mpi::broadcast(world, a, 0);
    boost::mpi::broadcast(world, b, 0);

    if (a == 0.0 && b == 0.0 && epsilon == 0.0) {
      throw std::runtime_error("Data was not broadcasted correctly.");
    }
  }

  return global_x_min;
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