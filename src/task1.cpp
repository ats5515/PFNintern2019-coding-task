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

	//ステップ数、特徴量の次元を入力
	int T, D;
	ifs >> T >> D;
	
	//重み行列の入力
	Matrix W;
	W.resize(D, Vector(D)); ifs >> W;


	//グラフの入力
	Graph g; 
	ifs >> g;

	//期待される出力ベクトルの入力
	Vector expected;
	expected.resize(D); ifs >> expected;

	assert(!ifs.eof());

	//モデル定義
	Functions *activation = new ReLU();
	GNN gnn(D, T, activation);
	gnn.W = W;

	//モデルの出力を得る
	Vector h_G = gnn.forward(g);
	
	//テストに合格かどうか
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

	//テストファイルのパス
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