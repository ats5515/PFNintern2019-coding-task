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

//�f�[�^�W�����Ǘ�����N���X


template<typename TX, typename TY>
class DataSet {
public:
	std::vector<TX> xs;//���f���ւ̓���
	std::vector<TY> ys;//���t�f�[�^
	DataSet() {

	}
	DataSet(std::string dirname, std::string x_suffix, std::string y_suffix = "") {
		assert(input(dirname, x_suffix, y_suffix));
	}

	int size() const {
		return (int)xs.size();
	}

	/*
	�f�[�^�̕���
	d�Ƀf�[�^��*rate���̃f�[�^�𕪊����Đݒ肷��B
	*/
	void split(DataSet<TX, TY> &d, double rate, std::mt19937 &mt) {
		d.xs.clear();	d.ys.clear();
		int num = (int)(xs.size() * rate + 0.01);
		assert(num > 0 && num < (int)xs.size());

		std::uniform_real_distribution<> dist(0.0, 1.0);
		while ((int)d.size() < num) {
			//0����xs.size()-1�܂ł̗����𐶐�
			int j = (int)((double)dist(mt) * xs.size());

			//j�Ԗڂ̃f�[�^��d�փR�s�[
			d.xs.push_back(xs[j]);
			d.ys.push_back(ys[j]);

			//j�Ԗڂ̃f�[�^���폜
			std::swap(xs[j], xs.back()); xs.pop_back();
			std::swap(ys[j], ys.back()); ys.pop_back();
		}
	}

	//�f�[�^�Z�b�g��ǂݍ���
	//dirname�F�f�[�^�Z�b�g������f�B���N�g���ւ̃p�X
	//x_suffix:���̓f�[�^�̐ڔ���
	//y_suffix:���t�f�[�^�̐ڔ����B�Ȃ���Ύw�肵�Ȃ��B
	bool input(std::string dirname, std::string x_suffix, std::string y_suffix = "") {

		std::cout << "loading dataset from " << dirname << std::endl;
		xs.clear();
		ys.clear();

		std::vector<std::string> datanames;//�ڔ�����x_suffix�Ɉ�v����t�@�C�����̃��X�g

		DIR* dp = opendir(dirname.c_str());

		if (dp != NULL)
		{
			struct dirent* dent;

			//�f�B���N�g�����̃t�@�C���𑖍�����B
			do {
				dent = readdir(dp);
				if (dent != NULL) {
					std::string s = dent->d_name;//�t�@�C�����擾

					//�t�@�C�����̐ڔ�����x_suffix�ƈ�v���邩
					if (s.size() >= x_suffix.size() && x_suffix == s.substr(s.size() - x_suffix.size(), x_suffix.size())) {
						datanames.push_back((std::string(dent->d_name)).substr(0, s.size() - x_suffix.size()));
					}

				}
			} while (dent != NULL);
			closedir(dp);

			xs.resize(datanames.size());

			if (y_suffix != "")ys.resize(datanames.size());

			//�擾�t�@�C����������ۂɃf�[�^�����
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