// Anikin Maksim 2025

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

bool cmp(const Pt& a, const Pt& b);

bool cw(const Pt& a, const Pt& b, const Pt& c);

bool ccw(const Pt& a, const Pt& b, const Pt& c);

void convex_hull(std::vector<Pt>& points);

bool test_data(std::vector<Pt> alg_out_, int case_);

void create_test_data(std::vector<Pt>& alg_in_, int case_);

void create_random_data(std::vector<Pt>& alg_in_, int count);

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