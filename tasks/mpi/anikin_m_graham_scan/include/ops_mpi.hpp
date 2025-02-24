// Anikin Maksim 2025

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace anikin_m_graham_scan_mpi {

struct pt {
  int x;
  int y;
  bool operator==(const pt& other) const { return x == other.x && y == other.y; }
};

bool cmp(const pt& a, const pt& b);

bool cw(const pt& a, const pt& b, const pt& c);

bool ccw(const pt& a, const pt& b, const pt& c);

void convex_hull(std::vector<pt>& points);

bool test_data(std::vector<pt> alg_out_, int case_);

void create_test_data(std::vector<pt>& alg_in_, int case_);

void create_random_data(std::vector<pt>& alg_in_, int count);

class TestTaskMPI : public ppc::core::Task {
 public:
  explicit TestTaskMPI(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<pt> data_;
  boost::mpi::communicator world_;
};

}  // namespace anikin_m_graham_scan_mpi