#include"FaceID.h"
#include"Global.h"

int main() {
	samples = GetSamples();//��ȡ����
	CalIntegralDiagrams();//������������ͼ
	Train();//ѵ������ �ڼ���������Ȩ���Լ���ǿ�������е�����������Ȩ��
}
