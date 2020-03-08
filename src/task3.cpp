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
#include "SGD.hpp"
#include "GradientDescent.hpp"
#include "MomentumSGD.hpp"



const int epoch = 100;

void train(DataSet<Graph, int> &dataset_train, DataSet<Graph, int> dataset_valid, Optimizer &opt, int seed) {
	const int D = 8;
	const int T = 2;

	
	//���f���̒�`
	Functions *func = new ReLU();
	GNN gnn(D, T, func);
	LinearClassifier ln(D);
	GNN_Classifier gnn_classifier(&gnn, &ln);

	//�����֐���`
	BinaryCrossEntropy loss;
	BinaryAccuracy acc;
	
	//������
	std::mt19937 mt(seed);
	gnn_classifier.initialize(mt);



	std::cout << std::fixed << std::setprecision(10);

	std::string modelName = "task3_" + opt.name;
	std::cerr << "model: " << modelName << std::endl;

	std::ofstream result(modelName + ".csv");//�o�̓t�@�C��

	//epoch��w�K
	for (int e = 0; e < epoch; e++) {
		gnn_classifier.learn(dataset_train, &opt, &loss);

		//�����Ɛ��𗦂̌v�Z
		std::vector<double> scores_train = gnn_classifier.evaluate(dataset_train, { &loss, &acc });
		std::vector<double> scores_valid = gnn_classifier.evaluate(dataset_valid, { &loss, &acc });

		//�W���o�͂֏o��
		std::cout << "epoch: " << Util::str(e + 1, 3) << ",loss_train: " << scores_train[0] << ", accuracy_train: " << scores_train[1]
			<< ",loss_valid: " << scores_valid[0] << ", accuracy_valid: " << scores_valid[1] << std::endl;

		//�t�@�C���֏o��
		result << scores_train[0] << "," << scores_train[1] << ","
			<< scores_valid[0] << "," << scores_valid[1] << std::endl;

	}

	delete func;
}

int main() {
	DataSet<Graph, int> dataset_train("datasets/train", "_graph.txt", "_label.txt");
	//DataSet<Graph, int> dataset(".\\src", "_graph.txt", "_label.txt");
	DataSet<Graph, int> dataset_valid;

	std::mt19937 mt(0);

	dataset_train.split(dataset_valid, 0.3, mt);

	{
		//SGD��p�����ꍇ
		SGD opt;
		opt.eps = 0.0001;
		opt.learning_rate = 0.0001;
		opt.batch_size = 10;

		train(dataset_train, dataset_valid, opt, 0);
	}

	{
		//MomentumSGD��p�����ꍇ
		MomentumSGD opt;
		opt.eps = 0.0001;
		opt.learning_rate = 0.0001;
		opt.momentum = 0.9;
		opt.batch_size = 10;

		train(dataset_train, dataset_valid, opt, 0);
	}

	return 0;
}
