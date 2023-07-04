#include <opencv2\opencv.hpp>
#include <iostream>
#include <opencv2/core/utils/logger.hpp>
#include <string>
#include<Imageprocess.h>
#include<WinSock2.h> 
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>
#include <iostream>
#include <sstream>
//#include<stdio.h>
//#include<stdlib.h>
//#include<WinSock2.h>  //WindowsSocket编程头文件
//#include<iostream>
//#include<cstring>
#pragma comment(lib,"ws2_32.lib")//链接ws2_32.lib库文件到此项目中
using namespace cv;
using namespace std;
double Image_clarity(cv::Mat& image) {
	cv::Mat kern1 = (Mat_<char>(2, 1) << -1, 1);
	cv::Mat kern2 = (Mat_<char>(1, 2) << -1, 1);
	cv::Mat engImg1, engImg2;
	filter2D(image, engImg1, CV_32F, kern1);
	filter2D(image, engImg2, CV_32F, kern2);
	cv::Mat resImg = engImg1.mul(engImg1) + engImg2.mul(engImg2);
	double OutValue = mean(resImg)[0];
	return OutValue;
}
//void tcp() 
//{
//
//	//================全局常量==================
//			 //创建缓冲区
//	const int BUF_SIZE = 2048;
//	//================全局变量==================
//	SOCKET sockSer, sockCli;
//	SOCKADDR_IN addrSer, addrCli; //address
//	int naddr = sizeof(SOCKADDR_IN);
//
//	char sendbuf[BUF_SIZE];
//	char inputbuf[BUF_SIZE];
//	char recvbuf[BUF_SIZE];
//	//================函数声明==================
//	int firing(); {
//		cout << "服务器启动" << endl;
//		//加载socket库
//		WSADATA wsadata;
//		if (WSAStartup(MAKEWORD(2, 2), &wsadata) != 0)
//		{
//			//输出出错信息
//			cout << "载入socket库失败!" << endl;
//			system("pause");
//		}
//		else {
//			cout << "载入socket库成功!" << endl;
//		}
//		//创建Soucket;
//		sockSer = socket(AF_INET, SOCK_STREAM, 0);
//		//描述协议族,INET属于ipv4；
//		//sock_stream创建套接字类型：tcp；
//		//0不指定协议，常用的协议有tcp、udp等
//
//		//初始化地址包
//		addrSer.sin_addr.s_addr = inet_addr("192.168.138.1");
//		addrSer.sin_family = AF_INET;
//		addrSer.sin_port = htons(1111);
//
//		//绑定Socket(bind)
//		bind(sockSer, (SOCKADDR*)&addrSer, sizeof(SOCKADDR));
//		//强制将SOCKADDR_INET转化成SOCKEADDR
//
//		//监听
//		while (true) {
//			cout << "开始连接!" << endl;
//			//监听连接请求;
//			listen(sockSer, 5);
//			//等待连接最大数：5
//
//			//接受连接
//			sockCli = accept(sockSer, (SOCKADDR*)&addrCli, &naddr);
//			if (sockCli != INVALID_SOCKET) {
//				while (true)
//				{
//					cout << "连接成功" << endl;
//					cout << "请输入要发送给客户端的信息：" << endl;
//					cin >> sendbuf;
//					send(sockCli, sendbuf, sizeof(sendbuf), 0);
//					//strcpy(sendbuf, "hello");
//					//send(sockCli, sendbuf, sizeof(sendbuf), 0);
//
//					//接收客户端发来信息
//					recv(sockCli, recvbuf, sizeof(recvbuf), 0);
//					cout << "客户端发来的信息：" << recvbuf << endl;
//				}
//
//			}
//			else
//			{
//				cout << "连接失败!" << endl;
//			}
//		}
//		closesocket(sockSer);
//		closesocket(sockCli);
//
//
//	}
//
//}


		
void CameraAlgorithm::Autofocus(cv::Mat& image)
{
	double current_clarity = Image_clarity(image);

}
bool Imageprocess::OpenCameraAndRecord(const std::string& url, cv::Mat& image)
{
	cv::VideoCapture capture;
	bool result = capture.open(url);

	//检查是否成功打开
	if (!capture.isOpened())
	{
		cout << "打开文摄像头失败，url:" << url << endl;
		return result;
	}

	//打印视频参数：宽、高、每秒传输帧数
	int videoWidth = capture.get(CAP_PROP_FRAME_WIDTH);
	int videoHeight = capture.get(CAP_PROP_FRAME_HEIGHT);
	int videoFps = capture.get(CAP_PROP_FPS);

	cout << "视频分辨率为:" << videoWidth << "乘以" << videoHeight << ",fps:" << videoFps << endl;
	int count = 0;
	namedWindow("当前视频帧", WINDOW_NORMAL);
	namedWindow("背景减图后区域", WINDOW_NORMAL);
	cv::Mat tempframe, currentframe, previousframe;
	cv::Mat frame;
	int framenum = 0;


	while (true)
	{
		//读取帧
		tempframe = capture.read(frame);
		capture >> frame;
		tempframe = frame;
		framenum++;

		if (framenum == 1)
		{
			cv::cvtColor(tempframe, previousframe, COLOR_BGR2GRAY);
		}
		else
		{
			//Imageprocess Mt;
			//cv::Mat dst1 = Mt.histogram_enhancement(tempframe);

			////dst1 = Mt.imgPreprocess(dst1);
			////cv::cvtColor(dst1, dst1, COLOR_BGR2GRAY);
			//std::pair <cv::Mat, cv::Mat> Image = Mt.image_segmentation(dst1);
			//cv::Mat image1 = Image.first;
			//cv::Mat image2 = Image.second;
			//image1 = Mt.imgPreprocess(image1, 3, 3,true);
			//imshow("图一", image1);
			//image2 = Mt.imgPreprocess(image2, 1, 1,false);
			//imshow("图二", image2);
			////Mt.alignImages(image1, image2);
			////Mt.Sift_detection(image1, image2);
			////将tempframe转为单通道灰度图，输出为currentFrame
			//cv::cvtColor(tempframe, currentframe, COLOR_BGR2GRAY);
			////pair<cv::Mat, cv::Mat> Image1 = Mt.image_segmentation(previousframe);
			//////做差求绝对值，输出为currentframe，即差分图像
			//cv::Mat dst;
			////absdiff(Image.second, Image1.second,dst);

			////

			//absdiff(currentframe, previousframe, currentframe);

			////遍历差分图像中点，将图像信息二值化，大于阈值20，为255，反之则为0，输出为currentframe
			////threshold(currentframe, currentframe, 20, 255.0, THRESH_BINARY);

			//cv::Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(10, 10));
			//morphologyEx(currentframe, currentframe, MORPH_OPEN, kernel, Point(-1, -1), 2);
			////imshow("背景减图后区域", currentframe);
			////cout << dst.channels() << endl;
			//// dst = Mt.imgPreprocess(dst);
			////double a = Mt.Image_rectangle(dst);

			//Ptr<BackgroundSubtractorMOG2> Model = createBackgroundSubtractorMOG2();
			//Model->setHistory(500);
			//Model->setVarThreshold(16);
			//Model->setDetectShadows(true);
			//cv::Mat mogframe;
			//Model->apply(tempframe, mogframe);
			////显示图像 
			//imshow("MOG", mogframe);
			
			imshow("当前视频帧", tempframe);



		}

		//把当前帧保存作为下一次处理的前一帧 
		cv::cvtColor(tempframe, previousframe, COLOR_BGR2GRAY);
		cv::waitKey(1);
	}

	capture.release();
	return result;
}

cv::Mat Imageprocess::RegionGrowSegment(cv::Mat& srcImage,cv::Mat& MaskImage, const int& ch1Thres, const int& ch1LowerBind, const int& ch1UpperBind, const int& ch2Thres) {
	if (srcImage.channels() == 3) {
		cv::cvtColor(srcImage, srcImage, COLOR_RGB2GRAY);
	}
	else {
		srcImage = srcImage.clone();
		
	}
	double maxValue;
	Point   maxIdx(0, 0);    // 最小值坐标，最大值坐标
	minMaxLoc(MaskImage, NULL, &maxValue, NULL, &maxIdx);
	Point pt = maxIdx;
	Point pToGrowing;                       //待生长点位置
	int pGrowValue = 0;                             //待生长点灰度值
	Scalar pSrcValue = 0;                               //生长起点灰度值
	Scalar pCurValue = 0;                               //当前生长点灰度值
	cv::Mat growImage = cv::Mat::zeros(srcImage.size(), CV_8UC1);   //创建一个空白区域，填充为黑色
	//生长方向顺序数据
	int DIR[8][2] = { {-1,-1}, {0,-1}, {1,-1}, {1,0}, {1,1}, {0,1}, {-1,1}, {-1,0} };
	vector<Point> growPtVector;                     //生长点栈
	growPtVector.push_back(pt);                         //将生长点压入栈中
	growImage.at<uchar>(pt.y, pt.x) = 255;              //标记生长点
	pSrcValue = srcImage.at<uchar>(pt.y, pt.x);         //记录生长点的灰度值

	while (!growPtVector.empty())                       //生长栈不为空则生长
	{
		Point pt = growPtVector.back();                       //取出一个生长点
		growPtVector.pop_back();

		//分别对八个方向上的点进行生长
		for (int i = 0; i < 8; ++i)
		{
			pToGrowing.x = pt.x + DIR[i][0];
			pToGrowing.y = pt.y + DIR[i][1];
			//检查是否是边缘点
			if (pToGrowing.x < 0 || pToGrowing.y < 0 ||
				pToGrowing.x >(srcImage.cols - 1) || (pToGrowing.y > srcImage.rows - 1))
				continue; // <-- Missing semicolon
			pGrowValue = growImage.at<uchar>(pToGrowing.y, pToGrowing.x);       //当前待生长点的灰度值
			pSrcValue = srcImage.at<uchar>(pt.y, pt.x);
			if (pGrowValue == 0)                    //如果标记点还没有被生长
			{
				pCurValue = srcImage.at<uchar>(pToGrowing.y, pToGrowing.x);
				if (pCurValue[0] <= ch1UpperBind && pCurValue[0] >= ch1LowerBind && pCurValue[0] <= maxValue + ch2Thres && pCurValue[0] >= maxValue - ch2Thres)
				{
					if (abs(pSrcValue[0] - pCurValue[0]) < ch1Thres)                   //在阈值范围内则生长
					{
						growImage.at<uchar>(pToGrowing.y, pToGrowing.x) = 255;      //标记为白色
						growPtVector.push_back(pToGrowing);                 //将下一个生长点压入栈中
					}
				}
			}
		}
	}
	imshow("growimage", growImage);
	srcImage.release();
	return growImage.clone();

}

void Imageprocess::gas_error_elimination(cv::Mat& src_image,cv::Mat& IRImage ,cv::Mat& mask_image)
{
	cv::Mat binary_image;
	if (src_image.channels() == 3) {
		cv::cvtColor(src_image, binary_image, COLOR_RGB2GRAY);
		threshold(binary_image, binary_image, 5, 255, THRESH_BINARY);
	}
	else {
		binary_image = src_image.clone();
	}
	
	if (IRImage.channels() == 1) {
		cv::cvtColor(IRImage, IRImage, COLOR_GRAY2RGB);
	}
	else {
		IRImage = IRImage.clone();
	}
	cout << IRImage.channels()<< endl;
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(binary_image, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE);
	if (contours.empty())
	{
		cout << "找不到有效轮廓！" << endl;
		return;
	}
	
	map<double, vector<vector<Point>>, greater<double>> Contour_Domain;
	cv::Mat temp_mask_image = cv::Mat::zeros(src_image.size(), src_image.type());
	vector<Point> Growing;                       //待生长点位置
	for (int i = 0; i < contours.size(); i++)
	{
		auto area = contourArea(contours[i]);
		Contour_Domain[area].emplace_back(contours[i]);
	}
	vector<vector<Point>> contours1;
	int cnt = 0;
	for (auto it = Contour_Domain.begin(); it != Contour_Domain.end() && cnt < 2; ++it) // 绘制面积前三的轮廓
	{
		for (auto vec : it->second)
		{
				contours1.push_back(vec);
		}
		cnt++;
		if (cnt >= 2) break;
	}
	
	for (int i = 0; i < contours1.size(); i++) {
		Moments M = moments(contours1[i]);
		Rect rect1 = boundingRect(contours1[i]);//最小正矩形
		rectangle(temp_mask_image, rect1, Scalar(255,255,255), -1, 8);
		int cX = static_cast<int>(M.m10 / M.m00);
		int cY = static_cast<int>(M.m01 / M.m00);
		Growing.push_back(Point(cX, cY));
		//drawContours(temp_mask_image, contours1, i, Scalar(255,255,255), -1);

	}
	bitwise_not(temp_mask_image, temp_mask_image);
	cv::cvtColor(temp_mask_image, temp_mask_image, COLOR_RGB2GRAY);
	cv::Mat mask_grow = cv::Mat::ones(temp_mask_image.rows + 2, temp_mask_image.cols + 2, temp_mask_image.type());
	temp_mask_image.copyTo(mask_grow(Rect(1, 1, temp_mask_image.cols, temp_mask_image.rows)));
	cv::imshow("mask_grow", mask_grow);
	Rect roi;
	cv::cvtColor(IRImage, IRImage, COLOR_RGB2GRAY);
	cv::Mat GrowImage = IRImage.clone();
	for (int x = 0; x < Growing.size(); x++) {
	 floodFill(GrowImage,mask_grow ,Growing[x], Scalar(255, 255, 255), &roi, Scalar(5, 5, 5), Scalar(5, 5, 5), FLOODFILL_FIXED_RANGE);
	}
	//Growing.clear();
	threshold(GrowImage, GrowImage, 254, 255, THRESH_BINARY);
	GrowImage = GrowImage / 255;
	bitwise_not(temp_mask_image, temp_mask_image);
	temp_mask_image = temp_mask_image / 255;
	mask_image = IRImage.mul(GrowImage);
	mask_image = mask_image.mul(temp_mask_image);
	cv::imshow("mask_image", mask_image);
}

// 输入：capture - 视频捕获对象，pBackgroundKnn - KNN背景建模器指针
void Imageprocess::BackGroundKnn(cv::VideoCapture& capture, Ptr<BackgroundSubtractorKNN> pBackgroundKnn)
{
	const int HISTORY_NUM = 5; // 历史信息帧数
	const int nKNN = 3; // KNN聚类后判断为背景的阈值
	const float defaultDist2Threshold = 5.0f; // 灰度聚类阈值
	cv::Mat frame, FGMask, FGMask_KNN;
	std::vector<std::vector<unsigned char>> pixelHistoryGray; // 存储像素灰度历史
	std::vector<std::vector<unsigned char>> pixelHistoryIsBG; // 存储像素是否为背景的历史
	  int frameCnt = 0;
	pBackgroundKnn->setHistory(200);
	pBackgroundKnn->setDist2Threshold(600);
	pBackgroundKnn->setShadowThreshold(0.5);
	while (cv::waitKey(30) != 'q')
	{
		// 读取当前帧
		if (!capture.read(frame))
			exit(EXIT_FAILURE);
		cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
		if (pixelHistoryGray.empty())
		{
			// 初始化一些变量
			int rows = frame.rows;
			int cols = frame.cols;
			FGMask.create(rows, cols, CV_8UC1);
			// 分配内存并初始化历史记录
			pixelHistoryGray.resize(rows * cols, std::vector<unsigned char>(HISTORY_NUM, 0));
			pixelHistoryIsBG.resize(rows * cols, std::vector<unsigned char>(HISTORY_NUM, 0));
		}
		// 处理图像前景检测
		FGMask.setTo(cv::Scalar(255));
		// 将矩阵转换为一维向量以便于使用向量运算
		cv::Mat pixelValues = frame.reshape(1, 1);
		int nPixels = static_cast<int>(pixelValues.total());
		// 将历史记录图像转换为矩阵，以便于使用OpenCV库函数进行灰度聚类
		cv::Mat grayHistory(HISTORY_NUM, nPixels, CV_8UC1);
		for (int n = 0; n < HISTORY_NUM; n++)
		{
			uchar* src = pixelHistoryGray[n].data();
			uchar* dst = grayHistory.ptr(n);
			memcpy(dst, src, nPixels * sizeof(uchar));
		}
		// 进行灰度聚类
		std::vector<int> labels;
		cv::kmeans(grayHistory.t(), nKNN, labels, cv::TermCriteria(), 5, cv::KMEANS_PP_CENTERS);
		// 遍历每个像素并进行背景前景判断
		for (int i = 0; i < frame.rows; i++)
		{
			for (int j = 0; j < frame.cols; j++)
			{
				int fit = 0;
				int fit_bg = 0;
				int idx = i * frame.cols + j;
				// 比较确定前景/背景
				for (int n = 0; n < HISTORY_NUM; n++)
				{
					if (fabs(static_cast<float>(pixelValues.at<uchar>(idx)) - static_cast<float>(pixelHistoryGray[idx][n])) < defaultDist2Threshold)
					{
						fit++;
					}
					if (pixelHistoryIsBG[idx][n] == 1)
					{
						fit_bg++;
					}
				}
				// 判断当前像素是否为背景
				if (fit < HISTORY_NUM * 0.5f || fit_bg < HISTORY_NUM * 0.5f)
				{
					FGMask.at<unsigned char>(i, j) = 0;
					pixelHistoryIsBG[idx].back() = 1;
				}
				else
				{
					pixelHistoryIsBG[idx].back() = 0;
				}
					// 更新历史值
					int index = (frameCnt%HISTORY_NUM);
					pixelHistoryGray[idx][index] = pixelValues.at<uchar>(idx);
					pixelHistoryIsBG[idx][index] = pixelHistoryIsBG[idx][index - 1];
			}
		}
		// 对前景图像进行形态学操作并显示
		cv::morphologyEx(FGMask, FGMask, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3)));
		cv::imshow("FGMask", FGMask);
		// 使用KNN背景建模器并显示结果
		pBackgroundKnn->apply(frame, FGMask_KNN);
		cv::imshow("FGMask_KNN", FGMask_KNN);
		frameCnt++;
	}
	capture.release();
}

cv::Mat Regional_noise_cancellation(cv::Mat& image1, cv::Mat& image2) {
	cv::Mat fused_image1, fused_image2;
	threshold(image1, fused_image1, 0, 128, THRESH_OTSU);
	threshold(image2, fused_image2, 0, 128, THRESH_OTSU);
	//二值化
	if (image1.size() != image2.size() || image1.type() != image2.type()) {
		cout << "输入图像大小或类型不匹配" << endl;
		return cv::Mat(); // return an empty cv::Mat object
	}
	cv::Mat fused_image = fused_image1 + fused_image2;
	threshold(fused_image, fused_image, 130, 255, THRESH_BINARY);
	fused_image = fused_image / 255;
	cv::Mat dst_image;
	dst_image = image2.mul(fused_image);
	imshow("去噪图像", dst_image);

	return dst_image;
	
}

cv::Mat CameraAlgorithm::Image_color_enhancement(cv::Mat& image)
{
	Mat_<Vec3b>::iterator it = image.begin<Vec3b>();
	Mat_<Vec3b>::iterator itend = image.end<Vec3b>();
	while (it != itend)
	{
		int imgval1 = (*it)[0];
		int imgval2 = (*it)[1];
		int imgval3 = (*it)[2];
		if (imgval1 == 0 && imgval2 == 0 && imgval3 == 0) {
			(*it)[0] = (rand() % (158 + 1));
			(*it)[1] = (rand() % (158 + 1));
			(*it)[2] = 0;
		}

		it++;
	}
	cv::Mat ColorMap;
	applyColorMap(image, ColorMap, COLORMAP_JET);
	imshow("ColorMap", ColorMap);
	return image;
}
cv::Mat CameraAlgorithm::Image_selection(cv::Mat& image1, cv::Mat& image2)
{

	cv::Mat binary_image, binary_image1;

	if (image1.channels() == 3) {
		cv::cvtColor(image1, binary_image, COLOR_RGB2GRAY);
		cv::cvtColor(image2, binary_image1, COLOR_RGB2GRAY);
	}
	else {
		binary_image = image1.clone();
		binary_image1 = image2.clone();
	}
	threshold(binary_image, binary_image, 210, 255, THRESH_OTSU);
	threshold(binary_image1, binary_image1, 0, 255, THRESH_BINARY);
	vector<vector<Point>> contours, contours1;
	vector<Vec4i> hierarchy;
	findContours(binary_image, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE);
	findContours(binary_image1, contours1, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE);
	cv::Mat temp_mask_image = cv::Mat::zeros(image1.size(), image1.type());
	for (int i = 0; i < contours.size(); i++)
	{
		for (int j = 0; j < contours1.size(); j++)
		{
			double similarity = cv::matchShapes(contours[i], contours1[j], cv::CONTOURS_MATCH_I2, 0);
			if (similarity > 0) 
			{ // 阈值设为0.1，可根据实际情况调整
				std::vector<cv::Point> hull1, hull2;
				cv::convexHull(contours[i], hull1);
				cv::convexHull(contours1[j], hull2);
				bool inside1 = true, inside2 = true;
				for (const auto& point : contours[i]) {
					if (cv::pointPolygonTest(hull2, point, false) < 0) {
						inside1 = false;
						break;
					}
				}
				for (const auto& point : contours1[j]) {
					if (cv::pointPolygonTest(hull1, point, false) < 0) {
						inside2 = false;
						break;
					}
				}
				if (!inside1 && inside2) {
					cv::drawContours(temp_mask_image, contours, i, cv::Scalar(255,255,255), -1);
					// contour2 完全嵌套在 contour1 中
				}
			
			}
		}
	}
	imshow("temp_mask_image", temp_mask_image);
	return  temp_mask_image;
}

void drawEllipseWithBox(cv::Mat& image){

	vector<vector<Point> > contours;
	findContours(image, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
	RotatedRect box;
	for (int i = 0; i < contours.size(); i++) {
		vector<Point> p = contours[i];
		if (p.size() <= 5) {
			continue;
		}
		box = fitEllipse(p);
	}
	

	ellipse(image, box, Scalar(255), 3, LINE_AA);

}
cv::Mat CameraAlgorithm::ConnectNearRegion(cv::Mat& src)
{
	cv::Mat gray, imageBw;
	cv::cvtColor(src, gray, COLOR_BGR2GRAY);
	threshold(gray, imageBw, 0, 255, THRESH_BINARY);

	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(imageBw, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point());

	// 连接近距离的轮廓
	float distanceThreshold = 40.0; // 设定的近距离阈值
	vector<vector<Point>> connectedContours;

	for (int i = 0; i < contours.size(); ++i)
	{
		vector<Point> currentContour = contours[i];
		bool isConnected = false;

		for (int j = i + 1; j < contours.size(); ++j)
		{
			vector<Point> nextContour = contours[j];

			// 计算当前轮廓与下一个轮廓之间的最短距离
			double minDistance = DBL_MAX;
			for (const Point& point1 : currentContour)
			{
				for (const Point& point2 : nextContour)
				{
					double distance = norm(point1 - point2);
					if (distance < minDistance)
					{
						minDistance = distance;
					}
				}
			}

			if (minDistance < distanceThreshold)
			{
				// 连接轮廓
				currentContour.insert(currentContour.end(), nextContour.begin(), nextContour.end());
				isConnected = true;
				contours.erase(contours.begin() + j); // 移除已连接的轮廓
				--j; // 更新下标
			}
		}

		if (isConnected)
		{
			connectedContours.push_back(currentContour);
		}
		else
		{
			connectedContours.push_back(contours[i]);
		}
	}
	// 绘制连接后的轮廓
	cv::Mat connectedImage = cv::Mat::zeros(src.size(), CV_8UC3);
	for (const auto& contour : connectedContours)
	{
		drawContours(connectedImage, vector<vector<Point>>{contour}, -1, Scalar(255, 255, 255), -1);
	}


	cv::Mat element = getStructuringElement(MORPH_ELLIPSE, Size(20, 20));	//创建方形结构元
	morphologyEx(connectedImage, connectedImage, MORPH_CLOSE, element,Point(-1,-1), 2);	//形态学闭运算
	imshow("Connected Image", connectedImage);
	return connectedImage;
}
cv::Mat CameraAlgorithm::color_rendering(cv::Mat& image) {
	vector<cv::Mat> Image;
	vector<int> colorgrayscale = { 35,rand() % 63 + 33,rand() % 62 + 97,rand() % 64 + 160,rand() % 32 + 224 };
	int count = 1;
	cv::Mat dst = cv::Mat::zeros(image.size(), image.type());
	for (; count < 6; count++) {
		cv::Mat image1 = cv::Mat::zeros(image.size(), image.type());
		cv::Mat element = getStructuringElement(MORPH_ELLIPSE, Size(25 - 4 * count, 25 - 4 * count));
		dst = image.clone();
		erode(image, image, element);
		image1 = dst - image;
		threshold(image1, image1, 0, colorgrayscale[count - 1], THRESH_BINARY);
		Image.push_back(image1);
		if (count == 5) {
			threshold(image, image, 0, 255, THRESH_BINARY);
			Image.push_back(image);
		}
	}
	cv::Mat sum_image = cv::Mat::zeros(image.size(), image.type());
	for (const cv::Mat& img : Image) {
		add(sum_image, img, sum_image);
	}
	applyColorMap(sum_image, sum_image, COLORMAP_JET);
	return sum_image;
}
// 优化版本
// 输入：src-输入图像，mask-掩膜图像
// 输出：dst-输出图像
cv::Mat CameraAlgorithm::boxfilter(cv::Mat& src, cv::Mat& mask)
{
	//cv::cvtColor(mask, mask, COLOR_GRAY2BGR);
	CV_Assert(!src.empty() && !mask.empty());
	CV_Assert(src.size() == mask.size());
	CV_Assert(src.type() == mask.type());
	mask = mask / 255;
	cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(15, 15));
	cv::Mat image = mask.clone();
	cv::erode(mask, mask, element);
	cv::Mat image2;
	cv::multiply(src, (image - mask), image2);
	cv::medianBlur(image2, image2, 3);
	cv::multiply(src, mask, src);
	return src + image2;
}


double Gas_area(cv::Mat& image) {
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	double x = 0;
	findContours(image, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point());
	for (int i = 0; i < contours.size(); i++) {
		double b = contourArea(contours[i]);
		x += b;
	}
	return x;
}

bool CameraAlgorithm::image_zero(Mat image1, Mat image2) {
	bool ret1=false, ret2=false;
	cvtColor(image1, image1, COLOR_BGR2GRAY);
	bitwise_not(image2, image2);
	Mat channels[3];
	split(image2, channels);
	if (countNonZero(image1) == 0) { ret1 = true; }
	if (countNonZero(channels[0]) == 0||countNonZero(channels[1]) == 0||countNonZero(channels[2]) == 0) { ret2 = true; }
	if (ret1 && ret2) { return true; }
	else { return false; }
}


cv::Mat CameraAlgorithm::GetMask(cv::Mat image) {
	Mat Mask = Mat::zeros(image.size(), image.type());
	Mat Mask1 = Mat::zeros(image.size(), image.type());
	threshold(image, Mask, 80, 255, THRESH_BINARY_INV);
	threshold(image, Mask1, 10, 255, THRESH_BINARY);
	bitwise_and(Mask, Mask1, Mask);
	return Mask;
}