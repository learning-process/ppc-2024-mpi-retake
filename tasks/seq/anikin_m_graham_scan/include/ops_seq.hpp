// Anikin Maksim 2025

#include <vector>

#include "core/task/include/task.hpp"

namespace anikin_m_graham_scan_seq {

struct pt {
  double x, y;
};

bool cmp(pt a, pt b);

bool cw(pt a, pt b, pt c);

bool ccw(pt a, pt b, pt c);

void convex_hull(std::vector<pt>& a);

bool test_data(std::vector<pt> alg_out_, int case_);

void create_test_data(std::vector<pt>& alg_in_, int case_);

void create_random_data(std::vector<pt>& alg_in_, int count);

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<pt> data_;
};

} // namespace anikin_m_graham_scan_seq