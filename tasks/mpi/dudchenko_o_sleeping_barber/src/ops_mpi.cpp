#include "mpi/dudchenko_o_sleeping_barber/include/ops_mpi.hpp"

#include <boost/mpi/communicator.hpp>
#include <chrono>
#include <deque>
#include <stdexcept>
#include <thread>

namespace dudchenko_o_sleeping_barber_mpi {

bool TestSleepingBarber::PreProcessingImpl() {
  result_ = -1;

  if (task_data && !task_data->inputs_count.empty()) {
    max_wait_ = static_cast<int>(task_data->inputs_count[0]);
    return max_wait_ > 0;
  }

  return false;
}

bool TestSleepingBarber::ValidationImpl() {
  if (!task_data || task_data->outputs.empty() || task_data->outputs_count[0] != sizeof(int)) {
    return false;
  }

  return task_data && !task_data->inputs_count.empty() && task_data->inputs_count[0] > 1;
}

bool TestSleepingBarber::RunImpl() {
  int total_clients = 10;  // Примерное количество клиентов
  std::deque<int> waiting_clients;
  bool barber_busy = false;

  for (int client = 0; client < total_clients; ++client) {
    if (static_cast<int>(waiting_clients.size()) < max_wait_) {
      waiting_clients.push_back(client);
    }

    if (!barber_busy && !waiting_clients.empty()) {
      int next_client_id = waiting_clients.front();
      waiting_clients.pop_front();
      NextClient(next_client_id);
      barber_busy = true;
    }

    if (barber_busy && waiting_clients.empty()) {
      barber_busy = false;
    }
  }

  while (!waiting_clients.empty()) {
    int next_client_id = waiting_clients.front();
    waiting_clients.pop_front();
    NextClient(next_client_id);
  }

  result_ = 0;
  return true;
}

bool TestSleepingBarber::PostProcessingImpl() {
  *reinterpret_cast<int*>(task_data->outputs[0]) = result_;
  return true;
}

void TestSleepingBarber::NextClient(int client) {
  (void)client;
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
}

bool TestMPISleepingBarber::PreProcessingImpl() {
  result_ = -1;

  if (world_.rank() == 0) {
    max_wait_ = static_cast<int>(task_data->inputs_count[0]);
  }

  return true;
}

bool TestMPISleepingBarber::ValidationImpl() {
  if (world_.rank() == 0) {
    if (task_data->inputs_count.empty() || task_data->inputs_count[0] <= 0) {
      return false;
    }
  }

  if (world_.rank() < 0 || world_.rank() >= world_.size()) {
    throw std::runtime_error("Invalid rank.");
  }

  return true;
}

bool TestMPISleepingBarber::RunImpl() {
  if (world_.size() < 3) {
    HandleSmallWorld();
    return true;
  }

  if (world_.rank() == 0) {
    HandleBarber();
  } else if (world_.rank() == 1) {
    HandleReceptionist();
  } else {
    HandleClient();
  }

  return true;
}

void TestMPISleepingBarber::HandleSmallWorld() {
  if (world_.rank() == 0) {
    result_ = 0;
  }
}

void TestMPISleepingBarber::HandleBarber() {
  while (true) {
    int client = -1;
    world_.recv(1, 0, client);

    if (client == -1) {
      result_ = 0;
      break;
    }

    NextClient(client);
  }
}

void TestMPISleepingBarber::HandleReceptionist() {
  std::deque<int> waiting_clients;
  max_wait_ = static_cast<int>(task_data->inputs_count[0]);
  int remaining_clients = world_.size() - 2;
  bool barber_busy = false;

  while (true) {
    ProcessIncomingClients(waiting_clients);
    AssignClientToBarber(waiting_clients, barber_busy);
    ProcessBarberSignal(barber_busy);

    if (ShouldTerminate(waiting_clients, remaining_clients, barber_busy)) {
      world_.send(0, 0, -1);
      break;
    }

    ProcessClientCompletion(remaining_clients);
  }
}

void TestMPISleepingBarber::ProcessIncomingClients(std::deque<int>& waiting_clients) {
  int client = -1;
  if (world_.iprobe(boost::mpi::any_source, 0)) {
    world_.recv(boost::mpi::any_source, 0, client);
    bool accepted = (static_cast<int>(waiting_clients.size()) < max_wait_);
    if (accepted) {
      waiting_clients.push_back(client);
    }
    world_.send(client, 1, accepted);
  }
}

void TestMPISleepingBarber::AssignClientToBarber(std::deque<int>& waiting_clients, bool& barber_busy) {
  if (!barber_busy && !waiting_clients.empty()) {
    int next_client = waiting_clients.front();
    waiting_clients.pop_front();
    world_.send(0, 0, next_client);
    barber_busy = true;
  }
}

void TestMPISleepingBarber::ProcessBarberSignal(bool& barber_busy) {
  if (world_.iprobe(0, 4)) {
    int barber_signal = 0;
    world_.recv(0, 4, barber_signal);
    barber_busy = false;
  }
}

bool TestMPISleepingBarber::ShouldTerminate(const std::deque<int>& waiting_clients, int remaining_clients,
                                            bool barber_busy) {
  return waiting_clients.empty() && remaining_clients == 0 && !barber_busy;
}

void TestMPISleepingBarber::ProcessClientCompletion(int& remaining_clients) {
  if (world_.iprobe(boost::mpi::any_source, 3)) {
    int done_signal = 0;
    world_.recv(boost::mpi::any_source, 3, done_signal);
    remaining_clients--;
  }
}

void TestMPISleepingBarber::HandleClient() {
  int client = world_.rank();
  bool accepted = false;

  world_.send(1, 0, client);
  world_.recv(1, 1, accepted);

  if (accepted) {
    world_.recv(0, 2, client);
  }

  world_.send(1, 3, client);
}

bool TestMPISleepingBarber::PostProcessingImpl() {
  world_.barrier();

  if (world_.rank() == 0) {
    if (!task_data->outputs.empty() && task_data->outputs_count[0] == sizeof(int)) {
      *reinterpret_cast<int*>(task_data->outputs[0]) = result_;
    } else {
      return false;
    }
  }

  return true;
}

void TestMPISleepingBarber::NextClient(int client) {
  std::this_thread::sleep_for(std::chrono::milliseconds(std::chrono::milliseconds(20)));
  world_.send(client, 2, client);
  world_.send(1, 4, client);
}

}  // namespace dudchenko_o_sleeping_barber_mpi