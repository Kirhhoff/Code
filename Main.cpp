#include"FaceID.h"
#include"Global.h"

int main() {
	GetSamples();//��ȡ����
	CalIntegralDiagrams();//������������ͼ
	InitialSomeVariable();
	GenerateFeatures();
	ifstream fin(classifierPathName.c_str());
	int cur = 0;
	while (cur<20)
		fin >> Factor[0][cur++];
	//for (int i = 0; i < cur; i++)
		//DrawRectangle(Factor[0][i], samples[0]);
	DrawRectangle(Factor[0][5], samples[0]);

	//Train();//ѵ������ �ڼ���������Ȩ���Լ���ǿ�������е�����������Ȩ��
}
