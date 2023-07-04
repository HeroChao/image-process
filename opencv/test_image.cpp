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
	Mc.GetMask(image);
	cv::waitKey(0);
	destroyAllWindows();
	return 0;

}