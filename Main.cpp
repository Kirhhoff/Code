#include"FaceID.h"

int main() {
	string posPathName="Code/pos/pos.txt"; //������·����
	string negPathName="Code/neg/neg.txt";//������·����
	//string* classifierPathName;//����ǿ�������е���������Ȩ��·��������
	Sample* samples;//��������

	samples = GetSamples(posPathName, negPathName);//��ȡ����
	CalIntegralDiagrams(samples);//������������ͼ
	Train(samples);//ѵ������ �ڼ���������Ȩ���Լ���ǿ�������е�����������Ȩ��
}
