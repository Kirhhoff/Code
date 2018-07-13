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
#include<ctime>
using namespace std;
using namespace cv;
#define MAP_ROWS 100
#define MAP_COLS 100
#define SAMPLE_NUM 0
#define HARD_CLASSIFIER_STAGES 1
#define MODEL_NUM 5
#define FEATURE_NUM 200000
#define __TP 1000
#define __TN 1000
typedef struct {
	int model;//�ĸ�����
	int factor;//��������
	int xSize;
	int ySize;
	int X;
	int Y;
} Feature;

typedef struct {
	Mat img;
	Mat integralDiagram;
	bool result;
	double weight;
} Sample;


/*
	------------����ģ���Ǽ�����ò-------------
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
Mat* GetSamples(string& pathName,bool*& results);//��������ͼ
void Train(Mat* samples,Mat* integralDiagrams);//ѵ��
Mat* CalIntegralDiagrams(Mat* samples);//���������Ļ���ͼ ������һ������
//Mat LoadSampleWeights(string& sampleWeightPathName);

