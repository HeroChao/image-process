#include <opencv2\opencv.hpp>
#include <iostream>
#include <opencv2/core/utils/logger.hpp>

using namespace std;
using namespace cv;

Ptr<cv::BackgroundSubtractorMOG2> mog2 = createBackgroundSubtractorMOG2();
queue<cv::Mat> DIFIRframeBuffer;

Mat element22 = getStructuringElement(MORPH_ELLIPSE, Size(2, 2));
Mat element33 = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
Mat element44 = getStructuringElement(MORPH_ELLIPSE, Size(4, 4));
Mat element55 = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
Mat element88 = getStructuringElement(MORPH_ELLIPSE, Size(8, 8));
Mat element1212 = getStructuringElement(MORPH_ELLIPSE, Size(12, 12));
Mat element2020 = getStructuringElement(MORPH_ELLIPSE, Size(20, 20));
Mat element4040 = getStructuringElement(MORPH_ELLIPSE, Size(40, 40));
Mat element8080 = getStructuringElement(MORPH_ELLIPSE, Size(80, 80));
Mat element1515 = getStructuringElement(MORPH_RECT, Size(15, 15));
class GasDetect
{
public:
	int getparameter() const {
		return Historical_difference_score;
	}
	
	static Mat Obtaining_differential_images(Mat& image) {
		//获得红外镜头图像的差分图像
		Mat IR_curr_frame, IR_prev_frame, IRDIF_frame;
		cv::cvtColor(image, IR_curr_frame, cv::COLOR_BGR2GRAY);
		DIFIRframeBuffer.push(IR_curr_frame);
		if (DIFIRframeBuffer.size() == 2) {
			IR_prev_frame = DIFIRframeBuffer.front();
			absdiff(IR_curr_frame, IR_prev_frame, IRDIF_frame);
			DIFIRframeBuffer.pop();
			IR_prev_frame.release();
			IR_curr_frame.release();
			return IRDIF_frame;
		}
		else {
			return IR_curr_frame;
		}

	}
	cv::Mat background_separation(cv::Mat& image, cv::Mat& image1) {
		cv::Mat Background_Image = cv::Mat::zeros(ValidWidth, ValidHeigth, CV_8U);
		//输入image为差分图像，image1为红外图像
		cv::Mat element = getStructuringElement(MORPH_ELLIPSE, Size(2, 2));
		erode(image, image, element);
		cv::Mat background_img=cv::Mat::zeros(image1.size(),image1.type());
		Scalar whitePixelCount = cv::sum(image);
		if (whitePixelCount[0] <= 100) {
			Historical_difference_score++;
		}
		else {
			Historical_difference_score = 0;
		}
		
		if (Historical_difference_score == 20) {
			Background_Image = image1.clone();
		}
		return Background_Image;
	}
	cv::Mat gas_histogram(cv::Mat& image) {
		//获取图像直方图
		cv::Mat hist;
		//需要计算的图像的通道，灰度图像为0，BGR图像需要指定B,G,R
		const int channels[] = { 0 };
		int dims = 1;//设置直方图维度
		const int histSize[] = { 256 }; //直方图每一个维度划分的柱条的数目
		//每一个维度取值范围
		float pranges[] = { 0, 255 };//取值区间
		const float* ranges[] = { pranges };
		calcHist(&image, 1, channels, cv::Mat(), hist, dims, histSize, ranges, true, false);//计算直方图
		int scale = 2;//每个小矩形宽度；
		int hist_height = 256;
		cv::Mat hist_img = cv::Mat::zeros(hist_height, 256 * scale, CV_8UC3); //创建一个黑底的8位的3通道图像，高256，宽256*2
		double max_val;
		minMaxLoc(hist, 0, &max_val, 0, 0);//计算直方图的最大像素值
		cout << image.size() << "and" << max_val << endl;
		//将像素的个数整合到 图像的最大范围内
		//遍历直方图得到的数据
		for (int i = 0; i < 256; i++)
		{
			float bin_val = hist.at<float>(i);   //遍历hist元素（注意hist中是float类型）
			int intensity = cvRound(bin_val * hist_height / max_val);  //绘制高度
			rectangle(hist_img, Point(i * scale, hist_height - 1), Point((i + 1) * scale - 1, hist_height - intensity), Scalar(255, 255, 255));//绘制直方图
		}

		imshow("直方图", hist_img);

	}
	void gas_error_elimination(Mat& src_image, Mat& mask_image)//2023.4.22去除屏幕上的杂点干扰
	{
		//根据图像轮廓大小来保留图像轮廓，以减小杂波干扰
		Mat binary_image;
		if (src_image.channels() == 3) {
			cvtColor(src_image, binary_image, COLOR_RGB2GRAY);
			threshold(binary_image, binary_image, 5, 255, THRESH_BINARY);
		}
		else {
			binary_image = src_image.clone();
		}
		Mat element = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
		morphologyEx(binary_image, binary_image, MORPH_OPEN, element);
		vector<vector<Point>> contours;
		vector<Vec4i> hierarchy;
		findContours(binary_image, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE);
		if (contours.empty())
		{
			return;
		}
		map<double, vector<vector<Point>>, greater<double>> Contour_Domain;
		Mat temp_mask_image = Mat::zeros(src_image.size(), src_image.type());
		for (int i = 0; i < contours.size(); i++)
		{
			auto area = contourArea(contours[i]);
			Contour_Domain[area].emplace_back(contours[i]);
		}
		vector<vector<Point>> contours1;
		int cnt = 0;
		for (auto it = Contour_Domain.begin(); it != Contour_Domain.end() && cnt < 5; ++it) // 绘制面积前三的轮廓
		{
			for (auto vec : it->second)
			{
				contours1.push_back(vec);
			}
			cnt++;
		}
		for (int i = 0; i < contours1.size(); i++) {
			drawContours(temp_mask_image, contours1, i, Scalar(255, 255, 255), -1);
		}
		temp_mask_image = temp_mask_image / 255;
		mask_image = src_image.mul(temp_mask_image);
		temp_mask_image.release();
		binary_image.release();
	}
	static Mat ConnectNearRegion(Mat& src, const float& disthr)
	{
		//连接图像轮廓，使气体图像连续；
		Mat imageBw;
		threshold(src, imageBw, 0, 255, THRESH_BINARY);

		vector<vector<Point>> contours;
		vector<Vec4i> hierarchy;
		findContours(imageBw, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point());

		// 连接近距离的轮廓
		float distanceThreshold = disthr; // 设定的近距离阈值
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
		Mat connectedImage = Mat::zeros(src.size(), CV_8UC1);
		for (const auto& contour : connectedContours)
		{
			drawContours(connectedImage, vector<vector<Point>>{contour}, -1, Scalar(255, 255, 255), -1);
		}
		Mat element = getStructuringElement(MORPH_ELLIPSE, Size(30, 30));
		Mat element1 = getStructuringElement(MORPH_ELLIPSE, Size(8, 8));
		morphologyEx(connectedImage, connectedImage, MORPH_CLOSE, element, Point(-1, -1), 2);
		dilate(connectedImage, connectedImage, element1);	//形态学闭运算
		imageBw.release();
		return connectedImage;
	}
	static void init_model(const int& history,const int& threshold,const bool& detectshadows) {
		/* 设置自定义参数history（默认值500）：表示背景模型需要记忆的帧数。较大的值可以更好地适应长期变化的背景，但也会增加计算量。
		varThreshold（默认值16）：表示像素与背景模型之间的方差阈值。较小的值会导致更多的像素被标记为前景，而较大的值则会更加保守。
		detectShadows（默认值true）：指示算法是否应该检测并标记阴影。如果设置为true，则前景掩码中的阴影像素将被标记为灰色。
		shadowValue（默认值127）：指定阴影像素在前景掩码中的值。
		shadowThreshold（默认值0.5）：表示阴影像素的阈值。较小的值会导致更多的像素被标记为阴影。
		nmixtures（默认值5）：表示背景模型中高斯成分的数量。较大的值可以更好地适应复杂的背景，但也会增加计算量。*/
		mog2->setHistory(history);
		mog2->setVarThreshold(threshold);
		mog2->setDetectShadows(detectshadows);
		mog2->setShadowValue(127);
		mog2->setShadowThreshold(0.5);
		mog2->setNMixtures(5);
	}

	static cv::Mat IR_detected_algorithm(cv::Mat& image) {
		cv::Mat fgmask;
		//vector<vector<Point>> contours;
		//vector<Vec4i> hierarchy;
		fgmask = cv::Mat::zeros(image.size(), image.type());
		threshold(image, image, 15, 255, THRESH_BINARY);
		medianBlur(image, image, 3);
		morphologyEx(image, image, MORPH_OPEN, element55);
		/*findContours(image, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE);

		for (const auto& contour : contours)
		{
			drawContours(fgmask, vector<vector<Point>>{contour}, -1, Scalar(255, 255, 255), -1);
		}*/
		return image;
	}
	 static  cv::Mat gas_mask(cv::Mat& image1, cv::Mat& image2) {
		//输入红外图像image1，可见光图像image2。
		
		Mat GasMask,IRMask ,VISMask ,DifIRframe;
		//cvtColor(image2, image2, COLOR_BGR2GRAY);
		GasMask = cv::Mat::zeros(image1.size(), image1.type());
		IRMask = cv::Mat::zeros(image1.size(), image1.type());
		VISMask = cv::Mat::zeros(image1.size(), image1.type());

		DifIRframe = Obtaining_differential_images(image1);
		IRMask = IR_detected_algorithm(DifIRframe);
		mog2->apply(image2, VISMask, 0.1);
		threshold(VISMask, VISMask, 15, 255, THRESH_BINARY);
		morphologyEx(VISMask, VISMask, MORPH_OPEN, element33);
		dilate(VISMask, VISMask, element33);
		imshow("IRMask", IRMask);
		imshow("VISMask", VISMask);
		VISMask = VISMask / 255;
		multiply(IRMask, VISMask, GasMask);
		//add(IRMask, VISMask, GasMask);
		morphologyEx(GasMask, GasMask, MORPH_CLOSE, element55);
		dilate(GasMask, GasMask, element55);
		//GasMask = GasDetect::ConnectNearRegion(GasMask, 30);
		imshow("GasMask", GasMask);
		return GasMask;
	}
	void init_element() {
		
	}

protected:
	cv::Mat IR_curr_frame, VIS_curr_frame, IRDIF_frame, VISDIF_frame, IR_ROI, VIS_ROI;
	int Vis_Registra_X = 0, Vis_Registra_Y = 0, Vis_Registra_W = 0, Vis_Registra_H = 0;
	int IR_Registra_X = 0, IR_Registra_Y = 0, IR_Registra_W = 0, IR_Registra_H = 0;
	
	int Historical_difference_score = 0;
	int  ValidWidth = 1280;
	int ValidHeigth = 1024;
	string url1 = "rtsp://admin:zhgx1234@192.168.1.123:554/cam/realmonitor?channel=2&subtype=0&unicast=true";
	string url2 = "rtsp://admin:zhgx1234@192.168.1.123:554/cam/realmonitor?channel=1&subtype=0";
};

class DualOpticalThread:public GasDetect
{
public:
	cv::Mat IR_process() {
		background_separation(IRDIF_frame, IR_ROI);
	

	}
	cv::Mat VIS_process() {
	}
	cv::Mat GetT_Img() {
		IR_ROI.convertTo(T_Img, CV_32FC1, 0.011646645, -60.037760148);
	}
	cv::Mat gas_area_detection() {




	}

private:
	cv::Mat T_Img;
	

};
