#pragma once

#include <algorithm>
#include <iostream>
#include <vector>
#include <map>
#include <assert.h>
#include <random>
#include <sys/types.h>
#include <dirent.h>

#include "Matrix.hpp"

//データ集合を管理するクラス


template<typename TX, typename TY>
class DataSet {
public:
	std::vector<TX> xs;//モデルへの入力
	std::vector<TY> ys;//教師データ
	DataSet() {

	}
	DataSet(std::string dirname, std::string x_suffix, std::string y_suffix = "") {
		assert(input(dirname, x_suffix, y_suffix));
	}

	int size() const {
		return (int)xs.size();
	}

	/*
	データの分割
	dにデータ数*rate分のデータを分割して設定する。
	*/
	void split(DataSet<TX, TY> &d, double rate, std::mt19937 &mt) {
		d.xs.clear();	d.ys.clear();
		int num = (int)(xs.size() * rate + 0.01);
		assert(num > 0 && num < (int)xs.size());

		std::uniform_real_distribution<> dist(0.0, 1.0);
		while ((int)d.size() < num) {
			//0からxs.size()-1までの乱数を生成
			int j = (int)((double)dist(mt) * xs.size());

			//j番目のデータをdへコピー
			d.xs.push_back(xs[j]);
			d.ys.push_back(ys[j]);

			//j番目のデータを削除
			std::swap(xs[j], xs.back()); xs.pop_back();
			std::swap(ys[j], ys.back()); ys.pop_back();
		}
	}

	//データセットを読み込む
	//dirname：データセットがあるディレクトリへのパス
	//x_suffix:入力データの接尾辞
	//y_suffix:教師データの接尾辞。なければ指定しない。
	bool input(std::string dirname, std::string x_suffix, std::string y_suffix = "") {

		std::cout << "loading dataset from " << dirname << std::endl;
		xs.clear();
		ys.clear();

		std::vector<std::string> datanames;//接尾辞がx_suffixに一致するファイル名のリスト

		DIR* dp = opendir(dirname.c_str());

		if (dp != NULL)
		{
			struct dirent* dent;

			//ディレクトリ内のファイルを走査する。
			do {
				dent = readdir(dp);
				if (dent != NULL) {
					std::string s = dent->d_name;//ファイル名取得

					//ファイル名の接尾辞がx_suffixと一致するか
					if (s.size() >= x_suffix.size() && x_suffix == s.substr(s.size() - x_suffix.size(), x_suffix.size())) {
						datanames.push_back((std::string(dent->d_name)).substr(0, s.size() - x_suffix.size()));
					}

				}
			} while (dent != NULL);
			closedir(dp);

			xs.resize(datanames.size());

			if (y_suffix != "")ys.resize(datanames.size());

			//取得ファイル名から実際にデータを入力
			for (int idx = 0; idx < datanames.size(); idx++) {
				{
					std::ifstream ifs(dirname + "/" + datanames[idx] + x_suffix);
					assert(ifs.is_open());
					ifs >> xs[idx];
				}
				if (y_suffix != "") {
					std::ifstream ifs(dirname + "/" + datanames[idx] + y_suffix);
					assert(ifs.is_open());
					ifs >> ys[idx];
				}
			}


		}
		else {
			std::cerr << "can not open directory " << dirname << std::endl;
			return false;
		}


		return true;
	}
};