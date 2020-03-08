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

//�萔������GNN

class GNN2 : public AbstractGNN{
private:
public:

	Functions *func;
	Matrix W;

	GNN2(int _D, int _T, Functions *_func){
		D = _D;
		T = _T;
		func = _func;
	}
	std::string getName() override {
		std::string name = "GNN2";
		name += "_" + func->name;
		name += "_D" + Util::str(D);
		name += "_T" + Util::str(T);
		return name;
	}
	virtual void initialize(std::mt19937 &mt) override {
		W.clear();
		//�s��T�C�Y��D*(D+1)�ƂȂ�B
		W.resize(D, Vector(D + 1));

		std::normal_distribution<> dist(0, 0.4);
		for (int i = 0; i < D; i++) {
			for (int j = 0; j < D + 1; j++) {
				W[i][j] = dist(mt);
			}
		}

	}

	virtual Vector forward(const Graph &g) override {

		int node_size = g.adj.size();


		//�m�[�h�̓����ʃx�N�g���̏�����
		std::vector<Vector> x(node_size, Vector(D, 0));
		//�ŏ��̗v�f�̂�1
		for (int i = 0; i < node_size; i++) {
			x[i][0] = 1.0;
		}

		for (int step = 0; step < T; step++) {
			std::vector<Vector> new_x(node_size, Vector(D, 0));
			for (int node_idx = 0; node_idx < node_size; node_idx++) {


				//�W��-1
				for (const int &adjacent_idx : g.adj[node_idx]) {//�אڃm�[�h�̗�
					for (int dimention = 0; dimention < D; dimention++) {
						new_x[node_idx][dimention] += x[adjacent_idx][dimention];
					}
				}
				new_x[node_idx].push_back(1.0);//�萔���𖖔��ɒǉ�

				//�W��-2
				new_x[node_idx] = matvecmul(W, new_x[node_idx]);
				func->apply(new_x[node_idx]);

			}
			swap(x, new_x);

		}
		Vector h_G(D, 0);

		//READOUT
		for (int dimention = 0; dimention < D; dimention++) {
			for (int node_idx = 0; node_idx < node_size; node_idx++) {

				h_G[dimention] += x[node_idx][dimention];
			}
		}
		return h_G;

	}

	std::vector<double*> collectParams() override {
		std::vector <double*> res;
		for (int i = 0; i < D; i++) {
			for (int j = 0; j < D + 1; j++) {
				res.push_back(&W[i][j]);
			}
		}
		return res;
	}

};
