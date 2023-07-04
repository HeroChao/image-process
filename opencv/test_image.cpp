#include<fstream>
#include <string>
#include<Imageprocess.h>
#include <ctime>
#include<AWB.h>
#include<IR_VIS.h>
#include <thread>
#include <mutex>
#include<future>




using namespace cv;
using namespace std;



void   Delay(int   time)//time*1000为秒数 
{
	clock_t   now = clock();

	while (clock() - now < time);
}

Mat image_contrast(Mat image) {
	Mat  clahe_img;
	//cvtColor(image, clahe_img, COLOR_GRAY2BGR);
	clahe_img = image.clone();
	vector<cv::Mat> channels(3);
	split(clahe_img, channels);
	Ptr<cv::CLAHE> clahe = createCLAHE();
	// 直方图的柱子高度大于计算后的ClipLimit的部分被裁剪掉，然后将其平均分配给整张直方图   
	clahe->setClipLimit(6.0); // (int)(4.*(8*8)/256)  
	clahe->setTilesGridSize(Size(5, 4)); // 将图像分为8*8块  
	for (int i = 0; i < 3; i++) {
		clahe->apply(channels[i], channels[i]);
	}
	merge(channels, clahe_img);
	Mat image_clahe;
	cvtColor(clahe_img, image_clahe, COLOR_BGR2GRAY);
	clahe_img.release();
	imshow("image_clahe", image_clahe);
	return image_clahe;
}
int main(int argc, char** argv)
{
	utils::logging::setLogLevel(utils::logging::LOG_LEVEL_SILENT);

	cv::Mat image = imread("D:/文件安装包/20.png");
	cv::Mat image1 = imread("D:/文件安装包/21.png");

	if (image.empty())
	{
		cout << "找不到图像" << endl;
		return -1;
	}
	//namedWindow("输入窗口", WINDOW_AUTOSIZE);
	imshow("图像显示", image);
	//imshow("图像显示1", image1);
	Imageprocess Mg;
	CameraAlgorithm Mc;
	GasDetect gas;
	IR_VIS iv;
	GasProcess gp;
	image_contrast(image);
	cv::waitKey(0);
	destroyAllWindows();
	return 0;

}