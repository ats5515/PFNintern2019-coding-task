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

//�œK����̒��ۃN���X
class Optimizer {
public:
	double eps = 0.0001;//���l�����̐ۓ�
	std::string name;//�œK����̖��O

	//�w�K
	//dataSize: �f�[�^�̃T�C�Y
	//F: �f�[�^�̃C���f�b�N�X���w�肷��Ƃ��̃f�[�^�ɑ΂��鑹�����Ԃ�֐��B
	//params: �œK���̑ΏۂƂȂ�p�����[�^�ւ̃|�C���^�̃��X�g
	//epoch: �w�K��
	virtual void learn(int dataSize, std::function<double(int)> F, std::vector<double*> params, int epoch) = 0;


	virtual ~Optimizer(){}
};