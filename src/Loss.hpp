#pragma once

#include <algorithm>
#include <iostream>
#include <vector>
#include <assert.h>
#include <random>

#include "Matrix.hpp"
#include "Optimizer.hpp"
#include "DataSet.hpp"

//損失関数の定義する


//損失関数の抽象クラス
//TOutput: 機械学習モデルの出力=損失関数損失関数への入力
//TY損失関数の出力
template<typename TOutput, typename TY>
class Loss {
public:
	virtual double getLoss(TOutput, TY) = 0;
};


//バイナリクロスエントロピー
class BinaryCrossEntropy : public Loss<double, int> {
	//sはシグモイド関数にかける前の値
	double getLoss(double s, int label) override {

		/*log(1 + exp(s))はs=100では精度15桁で100に等しく、
		s=1000でオーバーフローを起こす。

		s>=100ではlog(1 + exp(s))=sと近似できると判断
		*/
		if (label == 0) {
			if (s < 100) {
				return log(1 + exp(s));
			}
			else
			{
				return s;
			}
		}
		else {
			if (-s < 100) {
				return log(1 + exp(-s));
			}
			else
			{
				return -s;
			}
		}
	}
};

//正解率を得る関数も都合上損失関数として定義しておく。
class BinaryAccuracy : public Loss<double, int> {
	double getLoss(double s, int label) override {

		if ((s > 0) == (label == 1)) {
			return 1;
		}
		else {
			return 0;
		}
	}
};
