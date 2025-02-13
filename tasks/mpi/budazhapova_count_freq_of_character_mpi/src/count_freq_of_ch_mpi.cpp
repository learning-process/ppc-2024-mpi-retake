#include <algorithm>
#include <functional>
#include <iostream>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "mpi/budazhapova_e_count_freq_character/include/count_freq_character_mpi_header.hpp"

int budazhapova_e_count_freq_character_mpi::counting_freq(std::string str, char symb) {
	int resalt = 0;
	for (unsigned long i = 0; i < str.length(); i++) {
		if (str[i] == symb) {
			resalt++;
		}
	}
	return resalt;
}

bool budazhapova_e_count_freq_character_mpi::TestMPITaskSequential::pre_processing() {
	internal_order_test();
	input_ = std::string(reinterpret_cast<char*>(taskData->inputs[0]), taskData->inputs_count[0]);
	symb = *reinterpret_cast<char*>(taskData->inputs[1]);
	res = 0;
	return true;
}

bool budazhapova_e_count_freq_character_mpi::TestMPITaskSequential::validation() {
	internal_order_test();
	return taskData->outputs_count[0] == 1;
}

bool budazhapova_e_count_freq_character_mpi::TestMPITaskSequential::run() {
	internal_order_test();
	res = counting_freq(input_, symb);
	return true;
}

bool budazhapova_e_count_freq_character_mpi::TestMPITaskSequential::post_processing() {
	internal_order_test();
	reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
	return true;
}
bool budazhapova_e_count_freq_character_mpi::TestMPITaskParallel::pre_processing() {
	internal_order_test();
	int world_rank = world.rank();

	if (world_rank == 0) {
		input_ = std::string(reinterpret_cast<char*>(taskData->inputs[0]), taskData->inputs_count[0]);
		symb = *reinterpret_cast<char*>(taskData->inputs[1]);
	}
	return true;
}

bool budazhapova_e_count_freq_character_mpi::TestMPITaskParallel::validation() {
	internal_order_test();
	if (world.rank() == 0) {
		return taskData->outputs_count[0] == 1;
	}
	return true;
}

bool budazhapova_e_count_freq_character_mpi::TestMPITaskParallel::run() {
	internal_order_test();
	int world_rank = world.rank();
	int delta = 0;
	if (world_rank == 0) {
		delta = taskData->inputs_count[0] % world.size() == 0 ? taskData->inputs_count[0] / world.size()
			: taskData->inputs_count[0] / world.size() + 1;
	}

	boost::mpi::broadcast(world, delta, 0);
	boost::mpi::broadcast(world, symb, 0);

	if (world_rank == 0) {
		for (int proc = 1; proc < world.size(); proc++) {
			world.send(proc, 0, input_.data() + proc * delta, delta);
		}
	}
	local_input_.resize(delta);
	if (world_rank == 0) {
		local_input_ = std::string(input_.begin(), input_.begin() + delta);
	}
	else {
		world.recv(0, 0, local_input_.data(), delta);
	}
	local_res = counting_freq(local_input_, symb);
	boost::mpi::reduce(world, local_res, res, std::plus<>(), 0);
	return true;
}

bool budazhapova_e_count_freq_character_mpi::TestMPITaskParallel::post_processing() {
	internal_order_test();
	if (world.rank() == 0) {
		reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
	}
	return true;
}