#pragma once

#include <algorithm>
#include <iostream>
#include <vector>
#include <assert.h>
#include <random>
#include <random>
#include <functional>


#include "Matrix.hpp"
#include "DataSet.hpp"
#include "Model.hpp"

//最適化器の抽象クラス
class Optimizer {
public:
	double eps = 0.0001;//数値微分の摂動
	std::string name;//最適化器の名前

	//学習
	//dataSize: データのサイズ
	//F: データのインデックスを指定するとそのデータに対する損失が返る関数。
	//params: 最適化の対象となるパラメータへのポインタのリスト
	//epoch: 学習回数
	virtual void learn(int dataSize, std::function<double(int)> F, std::vector<double*> params, int epoch) = 0;


	virtual ~Optimizer(){}
};