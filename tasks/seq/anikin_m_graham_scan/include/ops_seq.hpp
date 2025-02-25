// Anikin Maksim 2025

#include <vector>

#include "core/task/include/task.hpp"

namespace anikin_m_graham_scan_seq {

struct Pt {
  double x, y;
};

bool cmp(Pt a, Pt b);

bool cw(Pt a, Pt b, Pt c);

bool ccw(Pt a, Pt b, Pt c);

void convex_hull(std::vector<Pt>& a);

bool test_data(std::vector<Pt> alg_out_, int case_);

void create_test_data(std::vector<Pt>& alg_in_, int case_);

void create_random_data(std::vector<Pt>& alg_in_, int count);

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<Pt> data_;
};

}  // namespace anikin_m_graham_scan_seq