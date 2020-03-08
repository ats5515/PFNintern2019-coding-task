#pragma once

#include <algorithm>
#include <iostream>
#include <vector>
#include <assert.h>
#include <random>

#include "Matrix.hpp"
#include "Optimizer.hpp"
#include "DataSet.hpp"

//�����֐��̒�`����


//�����֐��̒��ۃN���X
//TOutput: �@�B�w�K���f���̏o��=�����֐������֐��ւ̓���
//TY�����֐��̏o��
template<typename TOutput, typename TY>
class Loss {
public:
	virtual double getLoss(TOutput, TY) = 0;
};


//�o�C�i���N���X�G���g���s�[
class BinaryCrossEntropy : public Loss<double, int> {
	//s�̓V�O���C�h�֐��ɂ�����O�̒l
	double getLoss(double s, int label) override {

		/*log(1 + exp(s))��s=100�ł͐��x15����100�ɓ������A
		s=1000�ŃI�[�o�[�t���[���N�����B

		s>=100�ł�log(1 + exp(s))=s�Ƌߎ��ł���Ɣ��f
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

//���𗦂𓾂�֐����s���㑹���֐��Ƃ��Ē�`���Ă����B
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
