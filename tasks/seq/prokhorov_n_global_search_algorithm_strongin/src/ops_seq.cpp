#include "seq/prokhorov_n_global_search_algorithm_strongin/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <vector>

bool prokhorov_n_global_search_algorithm_strongin_seq::TestTaskSequential::PreProcessingImpl() {
  result_ = 0;
  a_ = *reinterpret_cast<double*>(task_data->inputs[0]);
  b_ = *reinterpret_cast<double*>(task_data->inputs[1]);
  return true;
}

bool prokhorov_n_global_search_algorithm_strongin_seq::TestTaskSequential::ValidationImpl() {
  return task_data->inputs_count[0] == 2 && task_data->outputs_count[0] == 1 && task_data->inputs.size() == 2 &&
         task_data->outputs.size() == 1;
}

bool prokhorov_n_global_search_algorithm_strongin_seq::TestTaskSequential::RunImpl() {
  double eps = 0.0001;
  double r = 2.0;
  result_ = StronginAlgorithm(a_, b_, eps, r, f_);
  return true;
}

double prokhorov_n_global_search_algorithm_strongin_seq::TestTaskSequential::StronginAlgorithm(
    double a, double b, double eps, double r, const std::function<double(double)>& f) {
  std::vector<double> points = {a, b};
  double lipsh = 0.0;
  int k = 2;

  while (true) {
    for (int i = 0; i < (k - 1); ++i) {
      double f_left = f(points[i]);
      double f_right = f(points[i + 1]);
      lipsh = std::max(lipsh, std::abs((f_right - f_left) / (points[i + 1] - points[i])));
    }

    double m = lipsh != 0 ? r * lipsh : 1.0;

    int s = 0;
    double max_ch = (m * (points[1] - points[0])) +
                    ((f(points[1]) - f(points[0])) * (f(points[1]) - f(points[0])) / (m * (points[1] - points[0]))) -
                    (2 * (f(points[1]) + f(points[0])));

    for (int i = 1; i < (k - 1); i++) {
      double f_left = f(points[i]);
      double f_right = f(points[i + 1]);
      double ch = (m * (points[i + 1] - points[i])) +
                  ((f_right - f_left) * (f_right - f_left) / (m * (points[i + 1] - points[i]))) -
                  (2 * (f_right + f_left));
      if (ch > max_ch) {
        s = i;
        max_ch = ch;
      }
    }

    double new_x = ((points[s] + points[s + 1]) / 2) - ((f(points[s + 1]) - f(points[s])) / (2 * m));

    if ((points[s + 1] - points[s]) <= eps) {
      return *std::ranges::min_element(points, [&f](double a, double b) { return f(a) < f(b); });
    }

    points.push_back(new_x);
    std::ranges::sort(points);
    k++;
  }
}

bool prokhorov_n_global_search_algorithm_strongin_seq::TestTaskSequential::PostProcessingImpl() {
  *reinterpret_cast<double*>(task_data->outputs[0]) = result_;
  return true;
}
