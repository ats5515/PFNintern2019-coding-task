#pragma once

#include <algorithm>
#include <iostream>
#include <vector>
#include <cassert>

#include "Matrix.hpp"

//関数群

//抽象クラス
struct Functions {
	std::string name;//関数名
	virtual void apply(double&) = 0;//破壊的な関数適用

	//ベクトルの要素すべてに関数適用
	void apply(Vector &vec) {
		for (double &v : vec) {
			apply(v);
		}
	}
	virtual ~Functions(){}

};

struct ReLU : Functions {
	ReLU() {
		name = "ReLU";
	}
	void apply(double &d) override {
		if (d < 0)d = 0;
	}
};

struct LReLU : Functions {
	LReLU() {
		name = "LReLU";
	}
	void apply(double &d) override {
		d = std::max(d, 0.01 * d);
	}
};

struct ELU : Functions {
	ELU() {
		name = "ELU";
	}
	void apply(double &d) override {
		if (d < 0)
			d = exp(d) - 1;
	}
};

struct sigmoid : Functions {
	sigmoid() {
		name = "sigmoid";
	}
	void apply(double &d) override {
		d = 1.0 / (1 + std::exp(-d));
	}
};

struct cosine : Functions {
	cosine() {
		name = "cosine";
	}
	void apply(double &d) override {
		d = std::cos(d);
	}
};