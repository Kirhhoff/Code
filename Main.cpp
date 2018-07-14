#include"FaceID.h"
#include"Global.h"

using namespace std;
int main() {

	LoadClassifier();
	GetSamples();//获取样本
	CalIntegralDiagrams();//计算样本积分图
	InitialSomeVariable();
	GenerateFeatures();

	ifstream fin(classifierPathName.c_str());
	ExpressNose(LoadAImage("Code/pos_100/0000_02176.jpg"));
	//DrawRectangle(Factor[0][5], samples[0]);
	//Train();//训练样本 期间会更新样本权重以及各强分类器中的弱分类器的权重
}
