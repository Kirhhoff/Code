#include"FaceID.h"
#include"Global.h"

using namespace std;
int main() {

	LoadClassifier();
	GetSamples();//��ȡ����
	CalIntegralDiagrams();//������������ͼ
	InitialSomeVariable();
	GenerateFeatures();

	ifstream fin(classifierPathName.c_str());
	ExpressNose(LoadAImage("Code/pos_100/0000_02176.jpg"));
	//DrawRectangle(Factor[0][5], samples[0]);
	//Train();//ѵ������ �ڼ���������Ȩ���Լ���ǿ�������е�����������Ȩ��
}
