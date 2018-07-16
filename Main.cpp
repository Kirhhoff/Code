#include"FaceID.h"
#include"Global.h"
using namespace cv;
using namespace std;
#ifdef TRAIN
int main() {

	GetSamples();//获取样本
	CalIntegralDiagrams();//计算样本积分图
	InitialSomeVariable();//初始化一部分变量
	GenerateFeatures();//生成特征
	Train();//训练样本 期间会更新样本权重以及各强分类器中的弱分类器的权重
	cin.get();
}
#endif // TRAIN
#ifdef USE
int main(int argc,char** argv) {
	Sample *initialSample, *compressedSample;
	LoadClassifier();
	for (int i = 1; i < argc; i++) {
		string imageName = argv[i];//获取文件路径名
		rotatedImagePathName = "Code/"+to_string(i);//生成输出的翻转文件路径名
		initialSample = LoadAImage(imageName.c_str());//读入图片
		compressedSample = Compress(initialSample);//进行压缩和灰度转化
		CalOneSampleIntegralDiagram(compressedSample);//计算图片的积分图
		CalSampleAllFeatureValues(compressedSample);//计算图片的不同特征特征值
		PredictResult();//进行预测
		int NO = 0;
		for (int i = 0; i < MAX_WEAK_CLASSIFIER_NUM_PER_HARD; i++)
			if (predictResult[i]) {//若这个弱分类器预测有人脸就把它记录下来
				sortedWeakFeatures[NO] = weakFeatures[i];
				sortedWeakFeatures[NO].rate = (double)(sampleFeatureValue[i] - weakFeatures[i].threshold) / (weakFeatures[i].maxSampleValue - weakFeatures[i].threshold);
				NO++;
			}
		sort(sortedWeakFeatures, sortedWeakFeatures + NO, [](Feature f1, Feature f2) {return f1.rate > f2.rate; });
		for (int i = 0; i < NO; i++)
			Rotate(sortedWeakFeatures[i], *initialSample);//根据各个分类器分别进行器官翻转
		delete initialSample;
		delete compressedSample;
	}
}
#endif // USE

