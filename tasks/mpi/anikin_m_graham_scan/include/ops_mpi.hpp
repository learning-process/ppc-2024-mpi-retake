// Anikin Maksim 2025
#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace anikin_m_graham_scan_mpi {

struct Pt {
  int x;
  int y;
  bool operator==(const Pt& other) const { return x == other.x && y == other.y; }
};

bool Cmp(const Pt& a, const Pt& b);

bool Cw(const Pt& a, const Pt& b, const Pt& c);

bool Ccw(const Pt& a, const Pt& b, const Pt& c);

void ConvexHull(std::vector<Pt>& points);

class TestTaskMPI : public ppc::core::Task {
 public:
  explicit TestTaskMPI(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<Pt> data_;
  boost::mpi::communicator world_;
};

}  // namespace anikin_m_graham_scan_mpi