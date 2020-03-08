#pragma once

#include <algorithm>
#include <iostream>
#include <vector>
#include <assert.h>
#include <random>
#include <functional>

#include "Matrix.hpp"
#include "DataSet.hpp"
#include "Model.hpp"
#include "Optimizer.hpp"
#include "Util.hpp"

//�œK����MomentumSGD�̎���
class MomentumSGD : public Optimizer {
public:
	double learning_rate = 0.0001;
	double momentum = 0.9;
	int batch_size = 10;
	std::mt19937 mt;

	MomentumSGD() {
		name = "MomentumSGD";
	}

	void update(std::vector<double> &dw, std::vector<double> &pending, std::vector<double*> &params, int &cnt) {
		for (int p = 0; p < params.size(); p++) {
			dw[p] = -learning_rate * (pending[p] / cnt) + momentum * dw[p];
			pending[p] = 0;
			*params[p] += dw[p];
		}
		cnt = 0;
	}

	void learn(int dataSize, std::function<double(int)> F, std::vector<double*> params, int epoch) override {
		//cerr << params.size() << endl;
		std::vector<double> dw(params.size(), 0);//�O��̍X�V��
		std::vector<double> pending(params.size(), 0);//�����l�̒~��
		std::vector<int> order(dataSize);//�C���f�b�N�X�̃��X�g

		std::iota(order.begin(), order.end(), 0);


		int cnt = 0;
		for (int e = 0; e < epoch; e++) {
			Util::shuffle(order, mt);//�C���f�b�N�X�̃��X�g���V���b�t��
			
			for (int &dataIdx : order) {
				double base = F(dataIdx);
				for (int p = 0; p < params.size(); p++) {
					//���l����
					double tmp = *params[p];
					*params[p] += eps;
					pending[p] += (F(dataIdx) - base) / eps;
					*params[p] = tmp;
				}
				cnt++;
				if (cnt >= batch_size) {
					//batch_size���ƂɃp�����[�^�̍X�V
					update(dw, pending, params, cnt);
				}

			}

			if (cnt > 0) {
				update(dw, pending, params, cnt);
			}
		}
	}
};