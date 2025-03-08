#pragma once

#include <memory>
#include <utility>

#include "core/task/include/task.hpp"

namespace dudchenko_o_sleeping_barber_seq {

class TestSleepingBarber : public ppc::core::Task {
 public:
  explicit TestSleepingBarber(std::shared_ptr<ppc::core::TaskData> task_data) : Task(std::move(task_data)) {}

  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  int max_wait_{};
  int result_{};

  static void NextClient(int client);
};
}  // namespace dudchenko_o_sleeping_barber_seq