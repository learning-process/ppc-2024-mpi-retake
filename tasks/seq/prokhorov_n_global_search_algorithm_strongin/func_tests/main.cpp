#include <gtest/gtest.h>

#include <cmath>
#include <functional>
#include <memory>
#include <numbers>

#include "core/task/include/task.hpp"
#include "seq/prokhorov_n_global_search_algorithm_strongin/include/ops_seq.hpp"

TEST(prokhorov_n_global_search_algorithm_strongin_seq, x_square) {
  double a = 0;
  double b = 10;
  std::function<double(double)> f = [](double x) { return x * x; };
  double answer = 0;
  double eps = 0.0001;

  prokhorov_n_global_search_algorithm_strongin_seq::TestTaskSequential task(std::make_shared<ppc::core::TaskData>(), f);

  double result =
      prokhorov_n_global_search_algorithm_strongin_seq::TestTaskSequential::StronginAlgorithm(a, b, eps, 2.0, f);

  EXPECT_NEAR(answer, result, eps);
}

TEST(prokhorov_n_global_search_algorithm_strongin_seq, sin) {
  double a = -std::numbers::pi;
  double b = std::numbers::pi;
  std::function<double(double)> f = [](double x) { return std::sin(x); };
  double eps = 0.1;
  double answer = -std::numbers::pi / 2;

  prokhorov_n_global_search_algorithm_strongin_seq::TestTaskSequential task(std::make_shared<ppc::core::TaskData>(), f);

  double result =
      prokhorov_n_global_search_algorithm_strongin_seq::TestTaskSequential::StronginAlgorithm(a, b, eps, 2.0, f);

  EXPECT_NEAR(answer, result, eps);
}

TEST(prokhorov_n_global_search_algorithm_strongin_seq, cos) {
  double a = 0;
  double b = 2 * std::numbers::pi;
  std::function<double(double)> f = [](double x) { return std::cos(x); };
  double eps = 0.1;
  double answer = std::numbers::pi;

  prokhorov_n_global_search_algorithm_strongin_seq::TestTaskSequential task(std::make_shared<ppc::core::TaskData>(), f);

  double result =
      prokhorov_n_global_search_algorithm_strongin_seq::TestTaskSequential::StronginAlgorithm(a, b, eps, 2.0, f);

  EXPECT_NEAR(answer, result, eps);
}

TEST(prokhorov_n_global_search_algorithm_strongin_seq, exp) {
  double a = -1;
  double b = 1;
  std::function<double(double)> f = [](double x) { return std::exp(x); };
  double eps = 0.0001;
  double answer = -1;

  prokhorov_n_global_search_algorithm_strongin_seq::TestTaskSequential task(std::make_shared<ppc::core::TaskData>(), f);

  double result =
      prokhorov_n_global_search_algorithm_strongin_seq::TestTaskSequential::StronginAlgorithm(a, b, eps, 2.0, f);

  EXPECT_NEAR(answer, result, eps);
}

TEST(prokhorov_n_global_search_algorithm_strongin_seq, absolute) {
  double a = -10;
  double b = 10;
  std::function<double(double)> f = [](double x) { return std::abs(x); };
  double eps = 0.0001;
  double answer = 0;

  prokhorov_n_global_search_algorithm_strongin_seq::TestTaskSequential task(std::make_shared<ppc::core::TaskData>(), f);

  double result =
      prokhorov_n_global_search_algorithm_strongin_seq::TestTaskSequential::StronginAlgorithm(a, b, eps, 2.0, f);

  EXPECT_NEAR(answer, result, eps);
}
