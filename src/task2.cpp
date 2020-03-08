#include <algorithm>
#include <iostream>
#include <fstream>
#include <vector>
#include <assert.h>
#include <random>
#include <iomanip>

#include "Matrix.hpp"
#include "Graph.hpp"
#include "GNN.hpp"
#include "GNN_Classfier.hpp"
#include "LinearClassifier.hpp"
#include "DataSet.hpp"
#include "Optimizer.hpp"
#include "GradientDescent.hpp"
#include "Util.hpp"
const int D = 8;
const int T = 2;
const int seed = 2;
const int epoch = 50;
int main() {
	DataSet<Graph, int> dataset("src", "_graph.txt", "_label.txt");
	
	//モデルの定義
	Functions *func = new ReLU();
	GNN gnn(D, T, func);
	LinearClassifier ln(D);
	GNN_Classifier gnn_classifier(&gnn, &ln);

	//学習器（勾配降下法）を定義
	Optimizer *opt = new GradientDescent();
	
	//初期化
	std::mt19937 mt(seed);
	gnn_classifier.initialize(mt);


	BinaryCrossEntropy loss;
	BinaryAccuracy acc;

	std::cout << std::fixed << std::setprecision(10);
	//epoch回学習
	for (int e = 0; e < epoch; e++) {
		//学習
		gnn_classifier.learn(dataset, opt, &loss);

		//評価
		std::vector<double> scores = gnn_classifier.evaluate(dataset, { &loss, &acc });
		std::cout << "epoch: " << e + 1 << ", loss: " << scores[0] << ", accuracy: " << scores[1] << std::endl;
	}


	delete func;
	delete opt;
	return 0;
}
