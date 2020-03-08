#pragma once

#include <algorithm>
#include <iostream>
#include <vector>
#include <assert.h>
#include <random>
#include <functional>

#include "Matrix.hpp"
#include "Optimizer.hpp"
#include "DataSet.hpp"
#include "Loss.hpp"

//�@�B�w�K���f���̒��ۃN���X�B
//TInput:���f���ւ̓��͂̌^
//TOutput:���f������̏o�͂̌^
template<typename TInput, typename TOutput>
class Model {

public:

	//������
	virtual void initialize(std::mt19937&) = 0;

	//���f���ւ̓��͂���o�͂��v�Z
	virtual TOutput forward(const TInput&) = 0;

	//�œK���̑ΏۂƂȂ�ϐ��ւ̃|�C���^���o��
	virtual std::vector<double*> collectParams(void) = 0;

	//dataset�ɑ΂��郂�f���̗\���l��Ԃ�
	template <typename TY>
	std::vector<TOutput> predict(const DataSet<TInput, TY> &dataset) {
		std::vector<TOutput> result((int)dataset.size());
		for (int dataIdx = 0; dataIdx < (int)dataset.size(); dataIdx++) {
			result[dataIdx] = forward(dataset.xs[dataIdx]);
		}
		return result;
	}

	//dataset�ɑ΂��郂�f���̕��ϑ�����Ԃ��B
	template <typename TY>
	std::vector<double> evaluate(const DataSet<TInput, TY> &dataset, std::vector<Loss<TOutput, TY>* > loss) {
		std::vector<double> result(loss.size(), 0);
		for (int dataIdx = 0; dataIdx < (int)dataset.size(); dataIdx++) {
			TOutput y_out = forward(dataset.xs[dataIdx]);
			for (int lossIdx = 0; lossIdx < loss.size(); lossIdx++) {
				result[lossIdx] += loss[lossIdx]->getLoss(y_out, dataset.ys[dataIdx]);
			}
		}
		for (int lossIdx = 0; lossIdx < (int)loss.size(); lossIdx++) {
			result[lossIdx] /= dataset.size();
		}
		return result;
	}

	//�w�K���s���B
	//dataset:�P���f�[�^
	//opt:�œK����
	//loss�����֐�
	//epoch:�G�|�b�N��
	template <typename TY>
	void learn(const DataSet<TInput, TY> &dataset, Optimizer *opt, Loss<TOutput, TY> *loss, int epoch = 1) {

		//�����͍œK����֓�����B
		opt->learn(
			dataset.xs.size(),
			[&](int dataIdx) {
			return loss->getLoss(forward(dataset.xs[dataIdx]), dataset.ys[dataIdx]);
		},
			collectParams(),
			epoch
			);


	}

};

