#include"FaceID.h"
#include"Global.h"

using namespace std;
int main() {
	samples = GetSamples();//��ȡ����
	CalIntegralDiagrams();//������������ͼ
	Train();//ѵ������ �ڼ���������Ȩ���Լ���ǿ�������е�����������Ȩ��
}
