#include <algorithm>
#include <iostream>
#include <fstream>
#include <vector>
#include <assert.h>
#include <random>

#include "Matrix.hpp"
#include "Graph.hpp"
#include "GNN.hpp"

bool testCase(std::string filename) {
	std::ifstream ifs(filename);
	assert(ifs.is_open());

	//�X�e�b�v���A�����ʂ̎��������
	int T, D;
	ifs >> T >> D;
	
	//�d�ݍs��̓���
	Matrix W;
	W.resize(D, Vector(D)); ifs >> W;


	//�O���t�̓���
	Graph g; 
	ifs >> g;

	//���҂����o�̓x�N�g���̓���
	Vector expected;
	expected.resize(D); ifs >> expected;

	assert(!ifs.eof());

	//���f����`
	Functions *activation = new ReLU();
	GNN gnn(D, T, activation);
	gnn.W = W;

	//���f���̏o�͂𓾂�
	Vector h_G = gnn.forward(g);
	
	//�e�X�g�ɍ��i���ǂ���
	bool pass = compareVec(h_G, expected);
	if (pass) {
		std::cout << "passed " << filename << std::endl;
	}
	else {
		std::cout << "ERROR " << filename << std::endl;
		std::cout << "returned : " << h_G << std::endl;
		std::cout << "expected : " << expected << std::endl;
	}


	delete activation;

	return pass;

}

int main() {

	//�e�X�g�t�@�C���̃p�X
	std::vector<std::string> testFiles = 
	{
		"src/test1.txt", 
		"src/test2.txt",
		"src/test3.txt",
		"src/test4.txt"
	};

	for (std::string name : testFiles) {
		testCase(name);
	}

	return 0;
}