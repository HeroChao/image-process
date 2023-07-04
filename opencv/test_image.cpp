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



void   Delay(int   time)//time*1000Ϊ���� 
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
	// ֱ��ͼ�����Ӹ߶ȴ��ڼ�����ClipLimit�Ĳ��ֱ��ü�����Ȼ����ƽ�����������ֱ��ͼ   
	clahe->setClipLimit(6.0); // (int)(4.*(8*8)/256)  
	clahe->setTilesGridSize(Size(5, 4)); // ��ͼ���Ϊ8*8��  
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

	cv::Mat image = imread("D:/�ļ���װ��/20.png");
	cv::Mat image1 = imread("D:/�ļ���װ��/21.png");

	if (image.empty())
	{
		cout << "�Ҳ���ͼ��" << endl;
		return -1;
	}
	//namedWindow("���봰��", WINDOW_AUTOSIZE);
	imshow("ͼ����ʾ", image);
	//imshow("ͼ����ʾ1", image1);
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