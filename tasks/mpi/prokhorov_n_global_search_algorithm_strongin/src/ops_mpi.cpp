#include "mpi/prokhorov_n_global_search_algorithm_strongin/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/mpi/collectives/broadcast.hpp>
#include <boost/mpi/collectives/gather.hpp>
#include <cmath>
#include <cstring>
#include <functional>
#include <vector>

bool prokhorov_n_global_search_algorithm_strongin_mpi::TestTaskSequential::PreProcessingImpl() {
  result_ = 0;
  a_ = *reinterpret_cast<double*>(task_data->inputs[0]);
  b_ = *reinterpret_cast<double*>(task_data->inputs[1]);
  return true;
}

bool prokhorov_n_global_search_algorithm_strongin_mpi::TestTaskSequential::ValidationImpl() {
  return task_data->inputs_count[0] == 2 && task_data->outputs_count[0] == 1 && task_data->inputs.size() == 2 &&
         task_data->outputs.size() == 1;
}

bool prokhorov_n_global_search_algorithm_strongin_mpi::TestTaskSequential::RunImpl() {
  double eps = 0.0001;
  double r = 2.0;
  result_ = StronginAlgorithm(a_, b_, eps, r, f_);
  return true;
}

double prokhorov_n_global_search_algorithm_strongin_mpi::TestTaskSequential::StronginAlgorithm(
    double a, double b, double eps, double r, const std::function<double(double*)>& f) {
  std::vector<double> points = {a, b};
  double lipsh = 0.0;
  int k = 2;

  while (true) {
    for (int i = 0; i < (k - 1); ++i) {
      double f_left = f(points.data() + i);
      double f_right = f(points.data() + i + 1);
      lipsh = std::max(lipsh, std::abs((f_right - f_left) / (points[i + 1] - points[i])));
    }

    double m = lipsh != 0 ? r * lipsh : 1.0;

    int s = 0;
    double max_ch = (m * (points[1] - points[0])) +
                    ((f(points.data() + 1) - f(points.data())) * (f(points.data() + 1) - f(points.data())) /
                     (m * (points[1] - points[0]))) -
                    (2 * (f(points.data() + 1) + f(points.data())));

    for (int i = 1; i < (k - 1); i++) {
      double f_left = f(points.data() + i);
      double f_right = f(points.data() + i + 1);
      double ch = (m * (points[i + 1] - points[i])) +
                  ((f_right - f_left) * (f_right - f_left) / (m * (points[i + 1] - points[i]))) -
                  (2 * (f_right + f_left));
      if (ch > max_ch) {
        s = i;
        max_ch = ch;
      }
    }

    double new_x = ((points[s] + points[s + 1]) / 2) - ((f(points.data() + s + 1) - f(points.data() + s)) / (2 * m));

    if ((points[s + 1] - points[s]) <= eps) {
      return *std::ranges::min_element(points, [&f](double a, double b) { return f(&a) < f(&b); });
    }

    points.push_back(new_x);
    std::ranges::sort(points);
    k++;
  }
}

bool prokhorov_n_global_search_algorithm_strongin_mpi::TestTaskSequential::PostProcessingImpl() {
  *reinterpret_cast<double*>(task_data->outputs[0]) = result_;
  return true;
}

bool prokhorov_n_global_search_algorithm_strongin_mpi::TestTaskParallel::PreProcessingImpl() {
  if (world_.rank() == 0) {
    result_ = 0;
    std::memcpy(&a_, task_data->inputs[0], sizeof(double));
    std::memcpy(&b_, task_data->inputs[1], sizeof(double));
  }
  return true;
}

bool prokhorov_n_global_search_algorithm_strongin_mpi::TestTaskParallel::ValidationImpl() {
  bool flag = world_.rank() != 0 ||
              (task_data->inputs_count.size() == 2 && task_data->inputs_count[0] == 1 &&
               task_data->inputs_count[1] == 1 && task_data->outputs_count.size() == 1 &&
               task_data->outputs_count[0] == 1 && task_data->inputs.size() == 2 && task_data->outputs.size() == 1);
  broadcast(world_, flag, 0);
  return flag;
}

double prokhorov_n_global_search_algorithm_strongin_mpi::TestTaskParallel::StronginAlgorithmParallel(
    double a, double b, const std::function<double(double*)>& f, double eps) {
  std::vector<double> points = {a, b};
  double lipsh = 0.0;
  double r = 2.0;
  int k = 2;

  while (true) {
    for (int i = 0; i < (k - 1); ++i) {
      double f_left = f(points.data() + i);
      double f_right = f(points.data() + i + 1);
      lipsh = std::max(lipsh, std::abs((f_right - f_left) / (points[i + 1] - points[i])));
    }

    double m = lipsh != 0 ? r * lipsh : 1.0;

    int s = 0;
    double max_ch = (m * (points[1] - points[0])) +
                    ((f(points.data() + 1) - f(points.data())) * (f(points.data() + 1) - f(points.data())) /
                     (m * (points[1] - points[0]))) -
                    (2 * (f(points.data() + 1) + f(points.data())));

    for (int i = 1; i < (k - 1); i++) {
      double f_left = f(points.data() + i);
      double f_right = f(points.data() + i + 1);
      double ch = (m * (points[i + 1] - points[i])) +
                  ((f_right - f_left) * (f_right - f_left) / (m * (points[i + 1] - points[i]))) -
                  (2 * (f_right + f_left));
      if (ch > max_ch) {
        s = i;
        max_ch = ch;
      }
    }

    double new_x = ((points[s] + points[s + 1]) / 2) - ((f(points.data() + s + 1) - f(points.data() + s)) / (2 * m));

    if ((points[s + 1] - points[s]) <= eps) {
      return *std::ranges::min_element(points, [&f](double a, double b) { return f(&a) < f(&b); });
    }

    points.push_back(new_x);
    std::ranges::sort(points);
    k++;
  }
}

bool prokhorov_n_global_search_algorithm_strongin_mpi::TestTaskParallel::RunImpl() {
  if (world_.size() == 1) {
    result_ = StronginAlgorithmParallel(a_, b_, f_, 0.0001);
    return true;
  }

  double segment = world_.rank() == 0 ? std::abs(b_ - a_) / world_.size() : 0;
  broadcast(world_, segment, 0);
  broadcast(world_, a_, 0);
  broadcast(world_, b_, 0);

  double local_left = a_ + (segment * world_.rank());
  double local_right = local_left + segment;

  double local_result = StronginAlgorithmParallel(local_left, local_right, f_, 0.0001);

  std::vector<double> local_answer(world_.size());
  boost::mpi::gather(world_, local_result, local_answer, 0);

  if (world_.rank() == 0) {
    result_ = *std::ranges::min_element(local_answer, [this](double a, double b) { return f_(&a) < f_(&b); });
  }

  return true;
}

bool prokhorov_n_global_search_algorithm_strongin_mpi::TestTaskParallel::PostProcessingImpl() {
  if (world_.rank() == 0) {
    std::memcpy(task_data->outputs[0], &result_, sizeof(double));
  }
  return true;
}
