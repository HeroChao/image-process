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
	Mc.GetMask(image);
	cv::waitKey(0);
	destroyAllWindows();
	return 0;

}