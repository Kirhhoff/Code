#include"FaceID.h"
#include"Global.h"

int main() {
	GetSamples();//��ȡ����
	CalIntegralDiagrams();//������������ͼ
	Train();//ѵ������ �ڼ���������Ȩ���Լ���ǿ�������е�����������Ȩ��
}
