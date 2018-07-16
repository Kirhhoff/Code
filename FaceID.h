#pragma once
#include<opencv2\core\core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2\imgproc\imgproc.hpp>
#include<iostream>
#include<fstream>
#include<array>
#include<vector>
#include <amp.h>
#include<amp_math.h>
#include <algorithm>
#include<ctime>
#include<stdlib.h> 
#include<string>
#include<iomanip>
#include<direct.h>
#include<io.h>
using namespace std;
using namespace cv;
#define Version_20
#define USE

#ifdef Version_20
#define MAP_ROWS 20
#define MAP_COLS 20
#endif // Version_20
#ifdef Version_100
#define MAP_ROWS 100
#define MAP_COLS 100
#endif // Version_100
#define SAMPLE_NUM 2000
#define HARD_CLASSIFIER_STAGES 1
#define MAX_WEAK_CLASSIFIER_NUM_PER_HARD 50
#define MODEL_NUM 5
#define FEATURE_NUM 1000000
#define __TP 1000
#define __TN 1000
#define FINAL_FEATURE_NUM 4

typedef struct {
	//以下是不随迭代变化的量
	int model;//哪个大类
	int factor;//缩放因子
	int xSize;//x方向大小
	int ySize;//y方向大小
	int X;//左上角的X坐标
	int Y;//左上角的Y坐标
	int Number;//在所有特征中的编号

	//以下是在每次迭代中值不同的量
	double eRate;//该特征在阈值下的错误率
	int threshold;//该特征的当前阈值
	int maxSampleValue;//所有样本中对该特征的最大特征值
	int p;//指示不等号的方向
	double rate;//可靠率
} Feature;
typedef struct {
	Mat img;//样本图片
	Mat integralDiagram;//样本的积分图
	bool result;//样本实际有无人脸
	double weight;//样本权重
} Sample;
typedef struct {
	bool key;
	int value;
	double weight;
} Key_Value;//用于样本特征值排序
typedef struct {
	double errorRate;
	int Number;
} ER_Number;//用于分类器错误率排序

/* 
	------------特征模板标记及其外貌-------------
	0:	(s,t)=(1,2)
		---------
		|*******|	
		---------
		|		|
		---------	

	1:	(s,t)=(2,1)
		---------
		|	|***|
		|	|***|
		---------
	
	2:	(s,t)=(1,3)
		---------
		|		|
		---------
		|*******|	
		---------
		|		|
		---------	

	3.	(s,t)=(3,1)
		-------------
		|	|***|	|
		|	|***|	|
		-------------

	4.	(s,t)=(2,2)
		---------
		|	|***|
		---------
		|***|	|
		---------
*/
ostream& operator<<(ostream& os, Feature& feature);
#ifdef TRAIN
void GetSamples();//读入样本图
void Train();//训练
Key_Value* CalFeatureValue(Feature& feature);//计算所有样本对一个特征的特征值
void CalIntegralDiagrams();//计算所有样本的积分图
void GenerateFeatures();//生成特征
void CalFeatureMinErrorRate();//计算该特征的取得最小错误率时的阈值以及此时对应的错误率
Feature& StoreClassifier(ofstream& fout, int& curWeakClassifierNum, int stage);//将选出的最优弱分类器存储下来
void UpdateSampleWeight(Feature& bestFeature);//一轮迭代后进行样本的权重更新
void InitialSomeVariable();//初始化一部分全局量
ofstream& operator<<(ofstream& fout, Feature& feature);//输出到文件
#endif // TRAIN
#ifdef USE
void DrawRectangle(Feature& feature, Sample& image);//测试用的函数
void Rotate(Feature& feature, Sample& sample);//对图片根据对应特征位置及大小进行旋转
void Rotate0(Feature& feature, Sample &sample);//0号模型的旋转方法
void Rotate1(Feature& feature, Sample &sample);//1号模型的旋转方法
void Rotate2(Feature& feature, Sample &sample);//2号模型的旋转方法
void Rotate3(Feature& feature, Sample &sample);//3号模型的旋转方法
void Rotate4(Feature& feature, Sample &sample);//4号模型的旋转方法
Sample* Compress(Sample* origin);//将原图压缩成20*20的图片
ifstream& operator>>(ifstream& fin, Feature& feature);
void LoadClassifier();//加载训练好的弱分类器
Sample* LoadAImage(string imagePathName);//加载一张图片
void CalOneSampleIntegralDiagram(Sample* sample);//计算输入图片的积分图
int CalSampleOneFeatureValue(Sample* sample, Feature& feature);//计算输入图片对某一特征的特征值
void CalSampleAllFeatureValues(Sample* sample);//计算输入图片对所有特征的特征值
void PredictResult();//结果预测
#endif USE


