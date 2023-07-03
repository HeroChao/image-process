#include <string>
#include <thread>
#include <mutex>
#include<GasDetect.h>


using namespace cv;
using namespace std;


#define MAXBUFFER 10

queue<cv::Mat> IRframeBuffer;
queue<cv::Mat> VISframeBuffer;

mutex IR_VISLock;
mutex Gaslock;
class IR_VIS {
public:
	void setvideoinput(const string& u, const string& v) {
		//设置红外与可见光视频流输入：红外 u，可见 v
		url1 = u;
		url2 = v;
	}

	void setparameter(int width, int heigth) {
		//设置显示图像大小参数
		ValidWidth = width;
		ValidHeigth = heigth;
	}
	void setmode(const int mode) {
		//设置图像显示模式
		Mode = mode;
	}
	void video_IR_frame_acquisition() {
		cv::VideoCapture capture1;
		capture1.open(url1);
		//检查是否成功打开
		if (!capture1.isOpened())
		{
			cout << "打开红外摄像头失败，url:" << url1 << endl;
		}
		cv::Mat frame1;
		while (true) {
			capture1.read(frame1);
			capture1 >> frame1;
			lock_guard<mutex> lock(IR_VISLock);
			while (IRframeBuffer.size() < MAXBUFFER) {
				IRframeBuffer.push(frame1);
				break;
			}
			cv::waitKey(10);
		}
		capture1.release();
	}
	void video_VIS_frame_acquisition() {
		cv::VideoCapture capture2;
		capture2.open(url2);
		//检查是否成功打开
		if (!capture2.isOpened())
		{
			cout << "打开可见光摄像头失败，url:" << url2 << endl;
		}
		Mat frame2;
		while (true)
		{
			capture2.read(frame2);
			capture2 >> frame2;
			lock_guard<mutex> lock(IR_VISLock);
			while (VISframeBuffer.size() < MAXBUFFER) {
				VISframeBuffer.push(frame2);
				break;
			}
			cv::waitKey(10);
		}
		capture2.release();
	}
	void initialize_video() {
		IR_VIS iv;
		thread thread_IR(&IR_VIS::video_IR_frame_acquisition, &iv);
		thread thread_VIS(&IR_VIS::video_VIS_frame_acquisition, &iv);
		while (true) {
			cv::Mat IRframe, VISframe, DIFIRframe;
			lock_guard<mutex> lock(IR_VISLock);
			if( !IRframeBuffer.empty()&&!VISframeBuffer.empty())
			{
					IRframe = IRframeBuffer.front();
					VISframe = VISframeBuffer.back();
					IRframeBuffer.pop();
					VISframeBuffer.pop();
			}
			if (!IRframe.empty() && !VISframe.empty() ) {
				IRframe = IRframe((Rect(220, 440, ValidWidth, ValidHeigth)));
				VISframe = VISframe((Rect(80, 500, ValidWidth, ValidHeigth)));
				//this->image_merge(VISframe, IRframe, Mode);
				Mat img = GasDetect::gas_mask(IRframe, VISframe);
				waitKey(1);
			}
		}
;		thread_IR.join();
		thread_VIS.detach();
	}
	
	virtual void image_merge(cv::Mat& image1, cv::Mat& image2, const int& mode) {};
protected:
	cv::Mat IR_curr_frame, VIS_curr_frame, IRDIF_frame, VISDIF_frame, IR_ROI, VIS_ROI;
	int  ValidWidth = 1280;
	int ValidHeigth = 1024;
	int Mode = 0;
private:
		string url1 = "rtsp://admin:zhgx1234@192.168.1.123:554/cam/realmonitor?channel=2&subtype=0&unicast=true";
		string url2 = "rtsp://admin:zhgx1234@192.168.1.123:554/cam/realmonitor?channel=1&subtype=0";

};

cv::Point2i startP;
void myMouseevent(int event, int x, int y, int flags, void* userdata) {

	//cv::imshow("image show", *(cv::Mat*)userdata);
	cv::Mat dst = (*(cv::Mat*)userdata).clone();
	//判断左键按下，记录起始点坐标
	if (event == EVENT_LBUTTONDOWN) {
		startP.x = x;
		startP.y = y;
	}
	//鼠标左键为按下状态，显示需要画的矩形，只能画出鼠标左下角移动的矩形
	if (flags == EVENT_FLAG_LBUTTON) {
		//计算需画的矩形
		Rect rect(startP.x, startP.y, x - startP.x, y - startP.y);
		//在图像上画出矩形
		rectangle(dst, rect, Scalar(0, 0, 250), 2, 4);
		cv::imshow("image show", dst);
	}
	//左键松开，在原图像画出矩形
	if (event == EVENT_LBUTTONUP) {
		Rect rect(startP.x, startP.y, x - startP.x, y - startP.y);
		rectangle(*(cv::Mat*)userdata, rect, Scalar(0, 0, 250), 2, 4);
		cv::imshow("image show", *(cv::Mat*)userdata);
	}

}
class GasProcess:public IR_VIS {
public:

	void mouseEventTest(cv::Mat& src) {
		namedWindow("image show", WINDOW_AUTOSIZE);
		cv::imshow("image show", src);
		setMouseCallback("image show", myMouseevent, &src);
		cv::waitKey(0);
	}
	cv::Mat IR_Tmap(cv::Mat& image) {
		cv::Mat temperatureImg;
		cv::applyColorMap(image, temperatureImg, COLORMAP_JET);
		return temperatureImg;
		cv::Mat  T_Img = cv::Mat::zeros(ValidWidth, ValidHeigth, CV_32FC1);
		image.convertTo(T_Img, CV_32FC1, 0.011646645, -60.037760148);

	}
	void image_merge(cv::Mat& image1, cv::Mat& image2, const int& mode)override {
		cv::Mat merge_image, T_img ;
		T_img = IR_Tmap(image2);
		if (mode == 0) {
			if (!image1.empty() && !image2.empty()) {
				cv::cvtColor(image2, image2, COLOR_GRAY2BGR);
				//vconcat(image1, image2, merge_image);  
				// 纵向合并
				cv::hconcat(image1, image2, merge_image);  
				// 横向合并
				cv::imshow("可见红外图", merge_image);
			}
		}
		else {
			if (!image1.empty() && !T_img.empty()) {
				//vconcat(image1, T_img, merge_image);  
				// 纵向合并
				cv::hconcat(image1, T_img, merge_image);  
				// 横向合并
				cv::imshow("可见温度图", merge_image);
			}
		}
	}
	cv::Mat draw_color_bar(cv::Mat& image,cv::Mat& mask) {
		ColorMap();

	}
private:
	void ColorMap()
	{
		cv::Mat MapImg = cv::Mat::zeros(ValidHeigth, thickness, CV_8UC1);
		for (int i = 0; i < ValidHeigth; i++)
		{
			int grayvalue = (int)floor((double)i / 2);
			for (int j = 0; j < thickness; j++)
			{
				MapImg.at<uchar>(i, j) = 255 - grayvalue;
			}
		}
		cv::applyColorMap(MapImg, ColorBarImg, COLORMAP_JET);
	}


private:
	int thickness = 10;
	cv::Mat ColorBarImg = cv::Mat::zeros(ValidHeigth, thickness, CV_8UC3);
	
	
};