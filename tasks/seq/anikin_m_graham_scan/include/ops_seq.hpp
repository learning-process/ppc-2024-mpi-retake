// Anikin Maksim 2025
#pragma once

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace anikin_m_graham_scan_seq {

struct Pt {
  double x, y;
};

bool Cmp(Pt a, Pt b);

bool Cw(Pt a, Pt b, Pt c);

bool Ccw(Pt a, Pt b, Pt c);

void ConvexHull(std::vector<Pt>& a);

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