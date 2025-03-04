#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace shpynov_n_readers_writers_mpi {
inline void Adder(std::vector<int> &a) {
  for (unsigned long i = 0; i < a.size(); i++) {
    a[i]++;
  }
}

enum Procedures : std::uint8_t { kWriteBegin, kWriteEnd, kReadBegin, kReadEnd };

inline Procedures Hasher(std::string const &in_string) {
  if (in_string == "kWriteBegin") {
    return kWriteBegin;
  }
  if (in_string == "kWriteEnd") {
    return kWriteEnd;
  }
  if (in_string == "kReadBegin") {
    return kReadBegin;
  }
  if (in_string == "kReadEnd") {
    return kReadEnd;
  }
  return kWriteBegin;
};

class TestTaskMPI : public ppc::core::Task {
 public:
  explicit TestTaskMPI(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<int> critical_resource_;
  std::vector<int> result_;
  boost::mpi::communicator world_;
};
}  // namespace shpynov_n_readers_writers_mpi