#include"FaceID.h"

Feature** Factor;//����ǿ�������и����������ĵ�ʱ����
int weakClassifierNum[HARD_CLASSIFIER_STAGES] = { 20 };
int minSquare[MODEL_NUM] = { 16,16,24,24,32 };
int s[MODEL_NUM] = { 1,2,1,3,2 };
int t[MODEL_NUM] = { 2,1,3,1,2 };
Sample* samples;//��������
Feature* Features;//��������
int featureNum=0;//ȫ��������
ER_Number* ERtable;
double curTP=(double)__TP / SAMPLE_NUM;
double curTN=(double)__TN / SAMPLE_NUM;
string classifierPathName = "Code/classifiers.txt";
#ifdef Version_100
string posPathName = "Code/pos_100/pos.txt";
string negPathName = "Code/neg_100/neg.txt";
#endif // Version_100
#ifdef Version_20
string posPathName = "Code/pos/pos.txt";
string negPathName = "Code/neg/neg.txt";
#endif




