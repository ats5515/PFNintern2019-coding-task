#pragma once

#include <algorithm>
#include <iostream>
#include <vector>
#include <assert.h>
#include <random>

//便利関数

namespace Util {


	//std::vector<T>の要素をO(N)でシャッフル(Nは入力長さ)
	template<typename T>
	void shuffle(std::vector<T> &vec, std::mt19937 &mt) {
		std::uniform_real_distribution<> dist(0.0, 1.0);
		for (int i = vec.size() - 1; i > 0; i--) {
			std::swap(vec[i], vec[(int)((double)dist(mt) * (i + 1))]);
		}
	}

	//最小桁数指定の整数->文字列変換
	std::string str(int i, int minDigits = 1) {
		assert(minDigits >= 0);

		std::string res;
		while (i > 0) {
			res.push_back((i % 10) + '0');
			i /= 10;
		}
		while ((int)res.size() < minDigits)res.push_back('0');

		reverse(res.begin(), res.end());
		return res;

	}


}