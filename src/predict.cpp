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
#include "GNN2.hpp"
#include "GNN_Classfier.hpp"
#include "LinearClassifier.hpp"
#include "DataSet.hpp"
#include "Optimizer.hpp"
#include "SGD.hpp"
#include "GradientDescent.hpp"
#include "MomentumSGD.hpp"
#include "Adam.hpp"



const int epoch = 200;

void train(DataSet<Graph, int> &dataset_train, DataSet<Graph, int> dataset_valid, GNN_Classifier &gnn_classifier, int seed) {
	std::mt19937 mt(seed);

	BinaryCrossEntropy loss;
	BinaryAccuracy acc;

	Adam opt;

	gnn_classifier.initialize(mt);

	std::cout << std::fixed << std::setprecision(10);
	for (int e = 0; e < epoch; e++) {
		gnn_classifier.learn(dataset_train, &opt, &loss);
		std::vector<double> scores_train = gnn_classifier.evaluate(dataset_train, { &loss, &acc });
		std::vector<double> scores_valid = gnn_classifier.evaluate(dataset_valid, { &loss, &acc });
		std::cout << "epoch: " << Util::str(e + 1, 3) << ",loss_train: " << scores_train[0] << ", accuracy_train: " << scores_train[1]
			<< ",loss_valid: " << scores_valid[0] << ", accuracy_valid: " << scores_valid[1] << std::endl;
	}


}

int main() {
	//trainデータ読み込み
	DataSet<Graph, int> dataset_train("datasets/train", "_graph.txt", "_label.txt");
	DataSet<Graph, int> dataset_valid;
	
	std::mt19937 mt(0);//データ分割のための乱数生成器
	dataset_train.split(dataset_valid, 0.3, mt);//データ分割



	//定数項付き cosine 
	int D = 8; int T = 3;
	Functions *func = new cosine();
	GNN2 gnn(D, T, func);
	LinearClassifier ln(D);
	GNN_Classifier gnn_classifier(&gnn, &ln);

	//学習
	train(dataset_train, dataset_valid, gnn_classifier, 0);
	
	//testデータ読み込み
	DataSet<Graph, int> dataset_test("datasets/test", "_graph.txt", "");
	std::vector<double> y_pred = gnn_classifier.predict(dataset_test);
	std::ofstream ofs("prediction.txt");
	for (int i = 0; i < y_pred.size(); i++) {
		ofs << (y_pred[i] > 0 ? 1 : 0) << std::endl;
	}


	delete func;


	return 0;
}
