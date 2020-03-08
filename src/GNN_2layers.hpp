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
#include "Util.hpp"

//集約ステップにおいて２層のパーセプトロンを使う。
//レポート課題には未使用


class GNN_2layers : public AbstractGNN{
private:
public:

	int H;
	Functions *func;
	Matrix W1;
	Matrix W2;

	GNN_2layers(int _D, int _H, int _T, Functions *_func){
		D = _D;
		H = _H;
		T = _T;
		func = _func;
	}
	std::string getName() override {
		std::string name = "GNN_2layers";
		name += "_" + func->name;
		name += "_D" + Util::str(D);
		name += "_T" + Util::str(T);
		name += "_H" + Util::str(H);
		return name;
	}
	virtual void initialize(std::mt19937 &mt) override {
		W1.clear();
		W1.resize(H, Vector(D));
		W2.clear();
		W2.resize(D, Vector(H));

		std::normal_distribution<> dist(0, 0.4);
		for (int i = 0; i < H; i++) {
			for (int j = 0; j < D; j++) {
				W1[i][j] = dist(mt);
			}
		}
		for (int i = 0; i < D; i++) {
			for (int j = 0; j < H; j++) {
				W2[i][j] = dist(mt);
			}
		}

	}

	virtual Vector forward(const Graph &g) override {

		int node_size = g.adj.size();
		std::vector<Vector> x(node_size, Vector(D, 0));

		for (int i = 0; i < node_size; i++) {
			x[i][0] = 1.0;
		}

		for (int step = 0; step < T; step++) {
			std::vector<Vector> new_x(node_size, Vector(D, 0));
			for (int node_idx = 0; node_idx < node_size; node_idx++) {

				for (int adjacent_idx : g.adj[node_idx]) {
					for (int dimention = 0; dimention < D; dimention++) {
						new_x[node_idx][dimention] += x[adjacent_idx][dimention];
					}
				}
				new_x[node_idx] = matvecmul(W1, new_x[node_idx]);
				func->apply(new_x[node_idx]);
				new_x[node_idx] = matvecmul(W2, new_x[node_idx]);
				func->apply(new_x[node_idx]);

			}
			swap(x, new_x);

		}
		Vector h_G(D, 0);
		for (int dimention = 0; dimention < D; dimention++) {
			for (int node_idx = 0; node_idx < node_size; node_idx++) {

				h_G[dimention] += x[node_idx][dimention];
			}
		}
		return h_G;

	}

	std::vector<double*> collectParams() override {
		std::vector <double*> res;
		for (int i = 0; i < H; i++) {
			for (int j = 0; j < D; j++) {
				res.push_back(&W1[i][j]);
			}
		}
		for (int i = 0; i < D; i++) {
			for (int j = 0; j < H; j++) {
				res.push_back(&W2[i][j]);
			}
		}
		return res;
	}

};
