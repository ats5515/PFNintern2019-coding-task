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

//機械学習モデルの抽象クラス。
//TInput:モデルへの入力の型
//TOutput:モデルからの出力の型
template<typename TInput, typename TOutput>
class Model {

public:

	//初期化
	virtual void initialize(std::mt19937&) = 0;

	//モデルへの入力から出力を計算
	virtual TOutput forward(const TInput&) = 0;

	//最適化の対象となる変数へのポインタを出力
	virtual std::vector<double*> collectParams(void) = 0;

	//datasetに対するモデルの予測値を返す
	template <typename TY>
	std::vector<TOutput> predict(const DataSet<TInput, TY> &dataset) {
		std::vector<TOutput> result((int)dataset.size());
		for (int dataIdx = 0; dataIdx < (int)dataset.size(); dataIdx++) {
			result[dataIdx] = forward(dataset.xs[dataIdx]);
		}
		return result;
	}

	//datasetに対するモデルの平均損失を返す。
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

	//学習を行う。
	//dataset:訓練データ
	//opt:最適化器
	//loss損失関数
	//epoch:エポック数
	template <typename TY>
	void learn(const DataSet<TInput, TY> &dataset, Optimizer *opt, Loss<TOutput, TY> *loss, int epoch = 1) {

		//処理は最適化器へ投げる。
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

