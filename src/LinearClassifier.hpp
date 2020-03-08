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

//���`���ފ�̎���
class LinearClassifier : public Model<Vector, double> {
private:
public:
	int D;//������
	Vector A;//�p�����[�^�x�N�g��
	double b;//�萔���̌W��

	LinearClassifier(int _D) {
		D = _D;
	}

	void initialize(std::mt19937 &mt) override {
		A.clear();
		A.resize(D);

		std::normal_distribution<> dist(0, 0.4);
		for (int i = 0; i < D; i++) {
			A[i] = dist(mt);
		}

		b = 0;

	}

	virtual double forward(const Vector &x) override {

		double result = dot(A, x) + b;

		return result;

	}

	std::vector<double*> collectParams() override {
		std::vector <double*> res(D + 1);
		for (int i = 0; i < D; i++)res[i] = &A[i];
		res[D] = &b;
		return res;
	}
};
