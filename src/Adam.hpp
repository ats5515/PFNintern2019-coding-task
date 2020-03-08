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

//最適化器Adamの実装

class Adam : public Optimizer {
public:
	double learning_rate = 0.001;
	double momentum = 0.9;
	double beta1 = 0.9;
	double beta2 = 0.999;
	int batch_size = 10;
	std::mt19937 mt;

	Adam() {
		name = "Adam";
	}

	void update(std::vector<double> &m, std::vector<double> &v, std::vector<double> &pending, std::vector<double*> &params, int &cnt) {
		static double beta1_t = 1;//pow(beta1, update呼び出し回数);
		static double beta2_t = 1;//pow(beta1, update呼び出し回数);

		beta1_t *= beta1;
		beta2_t *= beta2;

		for (int p = 0; p < params.size(); p++) {
			pending[p] /= cnt;
			m[p] = beta1 * m[p] + (1 - beta1) * pending[p];
			v[p] = beta2 * v[p] + (1 - beta2) * pending[p] * pending[p];
			pending[p] = 0;

			*params[p] += -learning_rate * (m[p] / (1 - beta1_t)) / (sqrt(1e-8 + (v[p] / (1 - beta2_t))));
		}

		cnt = 0;
	}

	void learn(int dataSize, std::function<double(int)> F, std::vector<double*> params, int epoch) override {
		//cerr << params.size() << endl;
		std::vector<double> m(params.size(), 0);//1次モーメント
		std::vector<double> v(params.size(), 0);//2次モーメント
		std::vector<double> pending(params.size(), 0);//微分値の蓄積
		std::vector<int> order(dataSize);

		std::iota(order.begin(), order.end(), 0);


		int cnt = 0;
		for (int e = 0; e < epoch; e++) {
			Util::shuffle(order, mt);
			/*for (int i = 0; i < order.size(); i++) {
				cerr << *order[i] << " ";
			}
			cerr << endl;*/


			for (int &dataIdx : order) {
				double base = F(dataIdx);
				//cerr << base << endl;
				for (int p = 0; p < params.size(); p++) {
					double tmp = *params[p];
					*params[p] += eps;
					pending[p] += (F(dataIdx) - base) / eps;
					*params[p] = tmp;
				}
				cnt++;
				if (cnt >= batch_size) {
					update(m, v, pending, params, cnt);
				}

			}

			if (cnt > 0) {
				update(m, v, pending, params, cnt);
			}
		}
		/*for (int i = 0; i < params.size(); i++) {
			cerr << *params[i] << " ";
		}
		cerr << endl;*/

	}
};