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

//勾配降下法の実装

class GradientDescent : public Optimizer{
public:
	double learning_rate = 0.001;//学習係数

	GradientDescent() {
		name = "GradientDescent";
	}

	void learn(int dataSize, std::function<double(int)> F, std::vector<double*> params, int epoch) override {
		std::vector<double> dw(params.size());
		for (int e = 0; e < epoch; e++) {

			for (int dataIdx = 0; dataIdx < dataSize; dataIdx++) {
				double base = F(dataIdx);
				for (int p = 0; p < params.size(); p++) {
					//数値微分
					double tmp = *params[p];
					*params[p] += eps;//パラメータを動かす
					dw[p] = (F(dataIdx) - base) / eps;//微分値の取得
					*params[p] = tmp;//パラメータを元に戻す
				}
				for (int p = 0; p < params.size(); p++) {
					*params[p] += -learning_rate * dw[p];//更新
				}

			}

		}

	}
};