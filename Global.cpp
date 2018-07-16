#include"FaceID.h"
int weakClassifierNum[HARD_CLASSIFIER_STAGES] = { MAX_WEAK_CLASSIFIER_NUM_PER_HARD };
int minSquare[MODEL_NUM] = { 3,3,4,4,5 };//最小特征窗口大小 防止过拟合
int s[MODEL_NUM] = { 1,2,1,3,2 };//这两个数组记录了各个
int t[MODEL_NUM] = { 2,1,3,1,2 };//模型的基本横纵比
string classifierPathName = "Code/classifiers.txt";
#ifdef TRAIN
Feature** Factor;//各级强分类器中各弱分类器的当时副本
Sample* samples;//样本数组
Feature* Features;//特征数组
int featureNum=0;//全部特征数
ER_Number* ERtable;//错误率表 用于排序
double curTP=(double)__TP / SAMPLE_NUM;//当前全部正样本的权重和
double curTN=(double)__TN / SAMPLE_NUM;//当前全部负样本的权重和
#endif // !TRAIN
#ifdef USE
double weakFactors[MAX_WEAK_CLASSIFIER_NUM_PER_HARD];//所有弱分类器的权重
Feature weakFeatures[MAX_WEAK_CLASSIFIER_NUM_PER_HARD];//所有弱分类器
Feature sortedWeakFeatures[MAX_WEAK_CLASSIFIER_NUM_PER_HARD];//认为有人脸的弱分类器
int sampleFeatureValue[MAX_WEAK_CLASSIFIER_NUM_PER_HARD];//样本对各特征的特征值
bool predictResult[MAX_WEAK_CLASSIFIER_NUM_PER_HARD];//各弱分类器的预测结果
double P=0;//有人脸的概率
int name = 0;//用于生成文件名
string rotatedImagePathName;//旋转后图片的路径名
#endif // USE


#ifdef Version_100
string posPathName = "Code/pos_100/pos.txt";
string negPathName = "Code/neg_100/neg.txt";
#endif // Version_100
#ifdef Version_20
string posPathName = "Code/pos/pos.txt";
string negPathName = "Code/neg/neg.txt";
#endif







