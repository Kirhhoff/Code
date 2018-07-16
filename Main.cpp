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
		cout << i << endl;
		string imageName = argv[i];
		rotatedImagePathName = to_string(i);
		initialSample = LoadAImage(imageName.c_str());
		compressedSample = Compress(initialSample);
		CalOneSampleIntegralDiagram(compressedSample);
		CalSampleAllFeatureValues(compressedSample);
		PredictResult();
		int NO = 0;
		for (int i = 0; i < MAX_WEAK_CLASSIFIER_NUM_PER_HARD; i++)
			if (predictResult[i]) {
				//if (weakFeatures[i].model == 3) {
					//cout << weakFeatures[i].threshold << " " << weakFeatures[i].p << " " << sampleFeatureValue[i] << " " << weakFeatures[i].maxSampleValue << endl;
					//Rotate(weakFeatures[i], *initialSample);
				//}
				sortedWeakFeatures[NO] = weakFeatures[i];
				sortedWeakFeatures[NO].rate = (double)(sampleFeatureValue[i] - weakFeatures[i].threshold) / (weakFeatures[i].maxSampleValue - weakFeatures[i].threshold);
				NO++;
			}
		sort(sortedWeakFeatures, sortedWeakFeatures + NO, [](Feature f1, Feature f2) {return f1.rate > f2.rate; });
		for (int i = 0; i < NO; i++)
			Rotate(sortedWeakFeatures[i], *initialSample);
		delete initialSample;
		delete compressedSample;
	}
	cin.get();
}
#endif // USE

