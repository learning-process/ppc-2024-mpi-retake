#include "mpi/Shpynov_N_reader_writer/include/readers_writers_mpi.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

class C_sem { //semaphore class
private:
  int signal;

public:
  C_sem(int sig) { signal = sig; }

  bool TryLock() {
    if (signal != 0) {
      signal--;
      return true;
    }
    return false;
  }
  void Lock() { signal--; }
  void Unlock() { signal++; }
  bool IsOnlyUser() { return signal == 1; }
  bool IsFree() { return signal == 0; }
};

bool Shpynov_N_readers_writers_mpi::TestTaskMPI::ValidationImpl() {
  if (world.rank() == 0) {
    if (task_data->inputs_count[0] != task_data->outputs_count[0])
      return false;
    if (task_data->inputs_count[0] <= 0)
      return false;
  }
  return true;
}

bool Shpynov_N_readers_writers_mpi::TestTaskMPI::PreProcessingImpl() {
  if (world.rank() == 0) {
    unsigned int input_size = task_data->inputs_count[0];
    auto *in_ptr = reinterpret_cast<int *>(task_data->inputs[0]);
    critical_resource = std::vector<int>(in_ptr, in_ptr + input_size);

    unsigned int output_size = task_data->outputs_count[0];
    result = std::vector<int>(output_size, 0);
  }
  return true;
}

enum procedures { WriteBegin, WriteEnd, ReadBegin, ReadEnd };

procedures hasher(std::string const &inString) {
  if (inString == "WriteBegin")
    return WriteBegin;
  if (inString == "WriteEnd")
    return WriteEnd;
  if (inString == "ReadBegin")
    return ReadBegin;
  if (inString == "ReadEnd")
    return ReadEnd;
  return WriteBegin;
};
bool Shpynov_N_readers_writers_mpi::TestTaskMPI::RunImpl() {
  C_sem mutex(1);
  C_sem writer(1);
  C_sem read_count(0);
  C_sem proc(world.size() - 1);

  if (world.rank() == 0) { // world represents monitor

    while (!proc.IsFree()) { // processing requests untill all threads been used at least once
      std::string procedure;
      boost::mpi::status stat;

      stat = world.recv(boost::mpi::any_source, 0, procedure);
      int sender_name = stat.source();
      std::vector<int> NewRes(critical_resource.size());
      switch (hasher(procedure)) {
      case WriteBegin:
        if (writer.TryLock()) {
          world.send(sender_name, 2, std::string("clear"));
          world.send(sender_name, 1, critical_resource);
        } else {
          world.send(sender_name, 2, std::string("wait"));
        }
        break;

      case WriteEnd:
        world.recv(sender_name, 3, NewRes);
        critical_resource = NewRes;
        writer.Unlock();
        proc.Lock();
        break;

      case ReadBegin:
        if (mutex.TryLock()) {
          read_count.Unlock();
          if (read_count.IsOnlyUser()) {
            if (writer.TryLock()) {
            } else {
              world.send(sender_name, 2, std::string("wait"));
              mutex.Unlock();
              continue;
            }
          }
          mutex.Unlock();
          world.send(sender_name, 2, std::string("clear"));
          world.send(sender_name, 1, critical_resource);
        } else
          world.send(sender_name, 2, std::string("wait"));
        break;

      case ReadEnd:
        mutex.Lock();
        read_count.Lock();
        if (read_count.IsFree()) {
          writer.Unlock();
        }
        mutex.Unlock();
        proc.Lock();
        break;

      default:
        break;
      }
    }
    result = critical_resource;
  } else if (world.rank() % 2 == 0) { // reader
    std::string resp = "wait";
    while (resp != "clear") {
      world.send(0, 0, std::string("ReadBegin"));
      world.recv(0, 2, resp);
    }
    world.recv(0, 1, critical_resource);
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    world.send(0, 0, std::string("ReadEnd"));
  } else { // writer
    std::string resp = "wait";
    while (resp != "clear") {
      world.send(0, 0, std::string("WriteBegin"));
      world.recv(0, 2, resp);
    }
    world.recv(0, 1, critical_resource);
    for (int i = 0; i < critical_resource.size(); i++) {
      critical_resource[i] += 1;
    }
    world.send(0, 3, critical_resource);
    world.send(0, 0, std::string("WriteEnd"));
  }
  world.barrier();
  return true;
}
bool Shpynov_N_readers_writers_mpi::TestTaskMPI::PostProcessingImpl() {
  if (world.rank() == 0) {
    auto *output = reinterpret_cast<int *>(task_data->outputs[0]);
    std::copy(result.begin(), result.end(), output);
    return true;
  }
  return true;
}