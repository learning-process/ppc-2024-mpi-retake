#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <deque>
#include <memory>
#include <utility>

#include "core/task/include/task.hpp"

namespace dudchenko_o_sleeping_barber_mpi {

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

class TestMPISleepingBarber : public ppc::core::Task {
 public:
  explicit TestMPISleepingBarber(std::shared_ptr<ppc::core::TaskData> task_data) : Task(std::move(task_data)) {}

  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  int max_wait_{};
  int result_{};
  boost::mpi::communicator world_;

  void NextClient(int client);
  void HandleSmallWorld();
  void HandleBarber();
  void HandleReceptionist();
  void ProcessIncomingClients(std::deque<int>& waiting_clients);
  void AssignClientToBarber(std::deque<int>& waiting_clients, bool& barber_busy);
  void ProcessBarberSignal(bool& barber_busy);
  static bool ShouldTerminate(const std::deque<int>& waiting_clients, int remaining_clients, bool barber_busy);
  void ProcessClientCompletion(int& remaining_clients);
  void HandleClient();
};
}  // namespace dudchenko_o_sleeping_barber_mpi