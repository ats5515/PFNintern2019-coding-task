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

//���z�~���@�̎���

class GradientDescent : public Optimizer{
public:
	double learning_rate = 0.001;//�w�K�W��

	GradientDescent() {
		name = "GradientDescent";
	}

	void learn(int dataSize, std::function<double(int)> F, std::vector<double*> params, int epoch) override {
		std::vector<double> dw(params.size());
		for (int e = 0; e < epoch; e++) {

			for (int dataIdx = 0; dataIdx < dataSize; dataIdx++) {
				double base = F(dataIdx);
				for (int p = 0; p < params.size(); p++) {
					//���l����
					double tmp = *params[p];
					*params[p] += eps;//�p�����[�^�𓮂���
					dw[p] = (F(dataIdx) - base) / eps;//�����l�̎擾
					*params[p] = tmp;//�p�����[�^�����ɖ߂�
				}
				for (int p = 0; p < params.size(); p++) {
					*params[p] += -learning_rate * dw[p];//�X�V
				}

			}

		}

	}
};