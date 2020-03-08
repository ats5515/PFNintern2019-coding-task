#pragma once

#include <algorithm>
#include <iostream>

#include <vector>
#include <assert.h>
#include <random>

#include "Graph.hpp"
#include "Matrix.hpp"
#include "Functions.hpp"
#include "Model.hpp"
#include "GNN.hpp"
#include "LinearClassifier.hpp"
#include "Util.hpp"

//GNNと線形分類器を合わせたモデル。

class GNN_Classifier : public Model<Graph, double> {
private:
public:
	AbstractGNN* gnn;
	LinearClassifier* ln;
	GNN_Classifier(AbstractGNN* _gnn, LinearClassifier* _ln) {
		gnn = _gnn;
		ln = _ln;

		assert(gnn->D == ln->D);//GNNと線形分類器のつなぎが正しいかチェック
	}

	std::string getName() {
		std::string name = gnn->getName();
		return name;
	}

	virtual void initialize(std::mt19937 &mt) override {
		
		gnn->initialize(mt);
		
		ln->initialize(mt);

	}

	virtual double forward(const Graph &g) override {

		double result = ln->forward(gnn->forward(g));

		return result;

	}

	std::vector<double*> collectParams() override {
		std::vector<double*> p1 = gnn->collectParams();
		std::vector<double*> p2 = ln->collectParams();
		p1.insert(p1.end(), p2.begin(), p2.end());
		return p1;
	}

};
