#include "mpi/Shpynov_N_reader_writer/include/readers_writers_mpi.hpp"

#include <algorithm>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/status.hpp>
#include <chrono>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

class CSem {  // semaphore class
 private:
  int signal_;

 public:
  CSem(int sig) { signal_ = sig; }

  bool TryLock() {
    if (signal_ != 0) {
      signal_--;
      return true;
    }
    return false;
  }
  void Lock() { signal_--; }
  void Unlock() { signal_++; }
  [[nodiscard]] bool IsOnlyUser() const { return signal_ == 1; }

  [[nodiscard]] bool IsFree() const { return signal_ == 0; }

  int CheckAnotherSem(CSem &writer, CSem &read_count) {
    if (this->TryLock()) {
      read_count.Unlock();
      if (read_count.IsOnlyUser()) {
        if (writer.TryLock()) {
        } else {
          this->Unlock();
          return 1;
        }
      }
      return 2;
    }
    return 0;
  }

  void UnlockIfFree(CSem &writer) const {
    if (this->IsFree()) {
      writer.Unlock();
    }
  }
};

bool shpynov_n_readers_writers_mpi::TestTaskMPI::ValidationImpl() {
  if (world_.rank() == 0) {
    if (task_data->inputs_count[0] != task_data->outputs_count[0]) {
      return false;
    }
    if (task_data->inputs_count[0] <= 0) {
      return false;
    }
  }
  return true;
}

bool shpynov_n_readers_writers_mpi::TestTaskMPI::PreProcessingImpl() {
  if (world_.rank() == 0) {
    unsigned int input_size = task_data->inputs_count[0];
    auto *in_ptr = reinterpret_cast<int *>(task_data->inputs[0]);
    critical_resource_ = std::vector<int>(in_ptr, in_ptr + input_size);

    unsigned int output_size = task_data->outputs_count[0];
    result_ = std::vector<int>(output_size, 0);
  }
  return true;
}

bool shpynov_n_readers_writers_mpi::TestTaskMPI::RunImpl() {
  CSem mutex(1);
  CSem writer(1);
  CSem read_count(0);
  CSem proc(world_.size() - 1);

  if (world_.rank() == 0) {  // world_ represents monitor

    while (!proc.IsFree()) {  // processing requests untill all threads been
                              // used at least once
      std::string procedure;
      boost::mpi::status stat;

      stat = world_.recv(boost::mpi::any_source, 0, procedure);
      int sender_name = stat.source();
      std::vector<int> new_res(critical_resource_.size());
      int tmp = 0;
      switch (shpynov_n_readers_writers_mpi::Hasher(procedure)) {
        case kWriteBegin:
          if (writer.TryLock()) {
            world_.send(sender_name, 2, std::string("clear"));
            world_.send(sender_name, 1, critical_resource_);
          } else {
            world_.send(sender_name, 2, std::string("wait"));
          }
          break;

        case kWriteEnd:
          world_.recv(sender_name, 3, new_res);
          critical_resource_ = new_res;
          writer.Unlock();
          proc.Lock();
          break;

        case kReadBegin:
          tmp = mutex.CheckAnotherSem(writer, read_count);
          if (tmp == 0) {
            world_.send(sender_name, 2, std::string("wait"));

          } else if (tmp == 2) {
            mutex.Unlock();
            world_.send(sender_name, 2, std::string("clear"));
            world_.send(sender_name, 1, critical_resource_);
          } else {
            world_.send(sender_name, 2, std::string("wait"));
            mutex.Unlock();
            continue;
          }
          break;

        case kReadEnd:
          mutex.Lock();
          read_count.Lock();
          read_count.UnlockIfFree(writer);
          mutex.Unlock();
          proc.Lock();
          break;

        default:
          break;
      }
    }
    result_ = critical_resource_;
  } else if (world_.rank() % 2 == 0) {  // reader
    std::string resp = "wait";
    while (resp != "clear") {
      world_.send(0, 0, std::string("kReadBegin"));
      world_.recv(0, 2, resp);
    }
    world_.recv(0, 1, critical_resource_);
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    world_.send(0, 0, std::string("kReadEnd"));
  } else {  // writer
    std::string resp = "wait";
    while (resp != "clear") {
      world_.send(0, 0, std::string("kWriteBegin"));
      world_.recv(0, 2, resp);
    }
    world_.recv(0, 1, critical_resource_);
    shpynov_n_readers_writers_mpi::Adder(critical_resource_);
    world_.send(0, 3, critical_resource_);
    world_.send(0, 0, std::string("kWriteEnd"));
  }
  world_.barrier();
  return true;
}
bool shpynov_n_readers_writers_mpi::TestTaskMPI::PostProcessingImpl() {
  if (world_.rank() == 0) {
    auto *output = reinterpret_cast<int *>(task_data->outputs[0]);
    std::ranges::copy(result_.begin(), result_.end(), output);
    return true;
  }
  return true;
}