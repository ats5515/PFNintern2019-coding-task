#pragma once

#include <algorithm>
#include <iostream>
#include <vector>
#include <cassert>

//行列、ベクトルの定義
//ベクトルはstd::vector<double>の別名とする。
//行列はstd::vector<std::vector<double> >の別名とする。


using Vector = std::vector<double>;
using Matrix = std::vector<Vector>;

//単位行列
Matrix ID(int N) {
	Matrix A(N, Vector(N, 0));
	for (int i = 0; i < N; ++i) {
		A[i][i] = 1;
	}
	return A;
}

//内積
double dot(const Vector &a, const Vector &b) {
	double ans = 0;
	for (int i = 0; i < a.size(); ++i) {
		ans += a[i] * b[i];
	}
	return ans;
}

//行列とベクトルの積
Vector matvecmul(const Matrix &A, const Vector &v) {
	Vector ans(A.size(), 0);
	for (int i = 0; i < A.size(); ++i)
		for (int j = 0; j < A[0].size(); ++j)
			ans[i] += A[i][j] * v[j];
	return ans;
}

//ベクトルの比較
//誤差epsまで許容
bool compareVec(Vector A, Vector B, double eps = 1e-9) {
	if (A.size() != B.size())return false;

	for (int i = 0; i < A.size(); i++) {
		if (std::abs(A[i] - B[i]) > eps) {
			return false;
		}
	}

	return true;
}

//以下、入出力のための関数

std::istream& operator>>(std::istream& is, Vector&v) {
	for (int i = 0; i < v.size(); i++) {
		is >> v[i];
	}
	return is;
}

std::ostream& operator<<(std::ostream& os, const Vector&v) {
	os << "[";
	for (int i = 0; i < v.size(); i++) {
		if (i > 0)os << ", ";
		os << v[i];
	}
	os << "]";
	return os;
}

std::istream& operator>>(std::istream& is, Matrix &m) {
	for (int i = 0; i < m.size(); i++) {
		is >> m[i];
	}
	return is;
}



std::ostream& operator<<(std::ostream& os, const Matrix &m) {
	os << "[" << std::endl;
	for (int i = 0; i < m.size(); i++) {
		if (i > 0)os << ", " << std::endl;
		os << m[i];
	}
	os << std::endl;
	os << "]";
	return os;
}

