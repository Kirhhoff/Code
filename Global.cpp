#include"FaceID.h"
int weakClassifierNum[HARD_CLASSIFIER_STAGES] = { MAX_WEAK_CLASSIFIER_NUM_PER_HARD };
int minSquare[MODEL_NUM] = { 3,3,4,4,5 };//��С�������ڴ�С ��ֹ�����
int s[MODEL_NUM] = { 1,2,1,3,2 };//�����������¼�˸���
int t[MODEL_NUM] = { 2,1,3,1,2 };//ģ�͵Ļ������ݱ�
string classifierPathName = "Code/classifiers.txt";
#ifdef TRAIN
Feature** Factor;//����ǿ�������и����������ĵ�ʱ����
Sample* samples;//��������
Feature* Features;//��������
int featureNum=0;//ȫ��������
ER_Number* ERtable;//�����ʱ� ��������
double curTP=(double)__TP / SAMPLE_NUM;//��ǰȫ����������Ȩ�غ�
double curTN=(double)__TN / SAMPLE_NUM;//��ǰȫ����������Ȩ�غ�
#endif // !TRAIN
#ifdef USE
double weakFactors[MAX_WEAK_CLASSIFIER_NUM_PER_HARD];//��������������Ȩ��
Feature weakFeatures[MAX_WEAK_CLASSIFIER_NUM_PER_HARD];//������������
Feature sortedWeakFeatures[MAX_WEAK_CLASSIFIER_NUM_PER_HARD];//��Ϊ����������������
int sampleFeatureValue[MAX_WEAK_CLASSIFIER_NUM_PER_HARD];//�����Ը�����������ֵ
bool predictResult[MAX_WEAK_CLASSIFIER_NUM_PER_HARD];//������������Ԥ����
double P=0;//�������ĸ���
int name = 0;//���������ļ���
string rotatedImagePathName;//��ת��ͼƬ��·����
#endif // USE


#ifdef Version_100
string posPathName = "Code/pos_100/pos.txt";
string negPathName = "Code/neg_100/neg.txt";
#endif // Version_100
#ifdef Version_20
string posPathName = "Code/pos/pos.txt";
string negPathName = "Code/neg/neg.txt";
#endif







