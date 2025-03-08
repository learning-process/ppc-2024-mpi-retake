#include "mpi/prokhorov_n_rectangular_integration/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/collectives/broadcast.hpp>
#include <boost/mpi/collectives/reduce.hpp>
#include <boost/mpi/communicator.hpp>
#include <cmath>
#include <functional>
#include <vector>

using namespace std::chrono_literals;

bool prokhorov_n_rectangular_integration_mpi::TestTaskSequential::PreProcessingImpl() {
  if (task_data->inputs.empty() || task_data->inputs_count[0] != 3) {
    return false;
  }

  auto* inputs = reinterpret_cast<double*>(task_data->inputs[0]);

  lower_bound_ = inputs[0];
  upper_bound_ = inputs[1];
  n_ = static_cast<int>(inputs[2]);

  if (lower_bound_ >= upper_bound_) {
    return false;
  }

  return n_ > 0;
}

bool prokhorov_n_rectangular_integration_mpi::TestTaskSequential::ValidationImpl() {
  if (task_data->inputs_count[0] != 3) {
    return false;
  }

  if (task_data->outputs_count[0] != 1) {
    return false;
  }

  auto* inputs = reinterpret_cast<double*>(task_data->inputs[0]);
  double lower_bound = inputs[0];
  double upper_bound = inputs[1];
  if (lower_bound >= upper_bound) {
    return false;
  }

  int n = static_cast<int>(inputs[2]);
  return n > 0;
}

bool prokhorov_n_rectangular_integration_mpi::TestTaskSequential::RunImpl() {
  result_ = Integrate(f_, lower_bound_, upper_bound_, n_);
  return true;
}

bool prokhorov_n_rectangular_integration_mpi::TestTaskSequential::PostProcessingImpl() {
  reinterpret_cast<double*>(task_data->outputs[0])[0] = result_;
  return true;
}

void prokhorov_n_rectangular_integration_mpi::TestTaskSequential::SetFunction(
    const std::function<double(double)>& func) {
  f_ = func;
}

double prokhorov_n_rectangular_integration_mpi::TestTaskSequential::Integrate(const std::function<double(double)>& f,
                                                                              double lower_bound, double upper_bound,
                                                                              int n) {
  double step = (upper_bound - lower_bound) / n;
  double area = 0.0;

  for (int i = 0; i < n; ++i) {
    double x = lower_bound + ((i + 0.5) * step);
    area += f(x) * step;
  }

  return area;
}

bool prokhorov_n_rectangular_integration_mpi::TestTaskMPI::PreProcessingImpl() {
  bool success = true;
  if (world_.rank() == 0) {
    if (task_data->inputs.empty() || task_data->inputs_count[0] != 3) {
      success = false;
    } else {
      auto* inputs = reinterpret_cast<double*>(task_data->inputs[0]);
      lower_bound_ = inputs[0];
      upper_bound_ = inputs[1];
      n_ = static_cast<int>(inputs[2]);

      if (lower_bound_ >= upper_bound_) {
        success = false;
      } else {
        success = n_ > 0;
      }
    }
  }

  return success;
}

bool prokhorov_n_rectangular_integration_mpi::TestTaskMPI::ValidationImpl() {
  if (world_.rank() == 0) {
    if (task_data->inputs_count[0] != 3) {
      return false;
    }

    if (task_data->outputs_count[0] != 1) {
      return false;
    }

    auto* inputs = reinterpret_cast<double*>(task_data->inputs[0]);
    double lower_bound = inputs[0];
    double upper_bound = inputs[1];
    if (lower_bound >= upper_bound) {
      return false;
    }

    int n = static_cast<int>(inputs[2]);
    return n > 0;
  }

  return true;
}

bool prokhorov_n_rectangular_integration_mpi::TestTaskMPI::RunImpl() {
  boost::mpi::broadcast(world_, lower_bound_, 0);
  boost::mpi::broadcast(world_, upper_bound_, 0);
  boost::mpi::broadcast(world_, n_, 0);

  result_ = ParallelIntegrate(f_, lower_bound_, upper_bound_, n_);
  return true;
}

bool prokhorov_n_rectangular_integration_mpi::TestTaskMPI::PostProcessingImpl() {
  if (world_.rank() == 0) {
    reinterpret_cast<double*>(task_data->outputs[0])[0] = result_;
  }
  return true;
}

void prokhorov_n_rectangular_integration_mpi::TestTaskMPI::SetFunction(const std::function<double(double)>& func) {
  f_ = func;
}

double prokhorov_n_rectangular_integration_mpi::TestTaskMPI::ParallelIntegrate(const std::function<double(double)>& f,
                                                                               double lower_bound, double upper_bound,
                                                                               int n) {
  int rank = world_.rank();
  int size = world_.size();

  double step = (upper_bound - lower_bound) / n;
  double local_area = 0.0;

  int local_n = n / size;
  int remainder = n % size;

  int start = (rank * local_n) + std::min(rank, remainder);
  int end = start + local_n + (rank < remainder ? 1 : 0);

  for (int i = start; i < end; ++i) {
    double x = lower_bound + ((i + 0.5) * step);
    local_area += f(x) * step;
  }

  double global_area = 0.0;
  boost::mpi::reduce(world_, local_area, global_area, std::plus<>(), 0);

  return global_area;
}