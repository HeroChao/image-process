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
		//��ú��⾵ͷͼ��Ĳ��ͼ��
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
		//����imageΪ���ͼ��image1Ϊ����ͼ��
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
		//��ȡͼ��ֱ��ͼ
		cv::Mat hist;
		//��Ҫ�����ͼ���ͨ�����Ҷ�ͼ��Ϊ0��BGRͼ����Ҫָ��B,G,R
		const int channels[] = { 0 };
		int dims = 1;//����ֱ��ͼά��
		const int histSize[] = { 256 }; //ֱ��ͼÿһ��ά�Ȼ��ֵ���������Ŀ
		//ÿһ��ά��ȡֵ��Χ
		float pranges[] = { 0, 255 };//ȡֵ����
		const float* ranges[] = { pranges };
		calcHist(&image, 1, channels, cv::Mat(), hist, dims, histSize, ranges, true, false);//����ֱ��ͼ
		int scale = 2;//ÿ��С���ο�ȣ�
		int hist_height = 256;
		cv::Mat hist_img = cv::Mat::zeros(hist_height, 256 * scale, CV_8UC3); //����һ���ڵ׵�8λ��3ͨ��ͼ�񣬸�256����256*2
		double max_val;
		minMaxLoc(hist, 0, &max_val, 0, 0);//����ֱ��ͼ���������ֵ
		cout << image.size() << "and" << max_val << endl;
		//�����صĸ������ϵ� ͼ������Χ��
		//����ֱ��ͼ�õ�������
		for (int i = 0; i < 256; i++)
		{
			float bin_val = hist.at<float>(i);   //����histԪ�أ�ע��hist����float���ͣ�
			int intensity = cvRound(bin_val * hist_height / max_val);  //���Ƹ߶�
			rectangle(hist_img, Point(i * scale, hist_height - 1), Point((i + 1) * scale - 1, hist_height - intensity), Scalar(255, 255, 255));//����ֱ��ͼ
		}

		imshow("ֱ��ͼ", hist_img);

	}
	void gas_error_elimination(Mat& src_image, Mat& mask_image)//2023.4.22ȥ����Ļ�ϵ��ӵ����
	{
		//����ͼ��������С������ͼ���������Լ�С�Ӳ�����
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
		for (auto it = Contour_Domain.begin(); it != Contour_Domain.end() && cnt < 5; ++it) // �������ǰ��������
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
		//����ͼ��������ʹ����ͼ��������
		Mat imageBw;
		threshold(src, imageBw, 0, 255, THRESH_BINARY);

		vector<vector<Point>> contours;
		vector<Vec4i> hierarchy;
		findContours(imageBw, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point());

		// ���ӽ����������
		float distanceThreshold = disthr; // �趨�Ľ�������ֵ
		vector<vector<Point>> connectedContours;
		for (int i = 0; i < contours.size(); ++i)
		{
			vector<Point> currentContour = contours[i];
			bool isConnected = false;

			for (int j = i + 1; j < contours.size(); ++j)
			{
				vector<Point> nextContour = contours[j];
				// ���㵱ǰ��������һ������֮�����̾���
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
					// ��������
					currentContour.insert(currentContour.end(), nextContour.begin(), nextContour.end());
					isConnected = true;
					contours.erase(contours.begin() + j); // �Ƴ������ӵ�����
					--j; // �����±�
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
		// �������Ӻ������
		Mat connectedImage = Mat::zeros(src.size(), CV_8UC1);
		for (const auto& contour : connectedContours)
		{
			drawContours(connectedImage, vector<vector<Point>>{contour}, -1, Scalar(255, 255, 255), -1);
		}
		Mat element = getStructuringElement(MORPH_ELLIPSE, Size(30, 30));
		Mat element1 = getStructuringElement(MORPH_ELLIPSE, Size(8, 8));
		morphologyEx(connectedImage, connectedImage, MORPH_CLOSE, element, Point(-1, -1), 2);
		dilate(connectedImage, connectedImage, element1);	//��̬ѧ������
		imageBw.release();
		return connectedImage;
	}
	static void init_model(const int& history,const int& threshold,const bool& detectshadows) {
		/* �����Զ������history��Ĭ��ֵ500������ʾ����ģ����Ҫ�����֡�����ϴ��ֵ���Ը��õ���Ӧ���ڱ仯�ı�������Ҳ�����Ӽ�������
		varThreshold��Ĭ��ֵ16������ʾ�����뱳��ģ��֮��ķ�����ֵ����С��ֵ�ᵼ�¸�������ر����Ϊǰ�������ϴ��ֵ�����ӱ��ء�
		detectShadows��Ĭ��ֵtrue����ָʾ�㷨�Ƿ�Ӧ�ü�Ⲣ�����Ӱ���������Ϊtrue����ǰ�������е���Ӱ���ؽ������Ϊ��ɫ��
		shadowValue��Ĭ��ֵ127����ָ����Ӱ������ǰ�������е�ֵ��
		shadowThreshold��Ĭ��ֵ0.5������ʾ��Ӱ���ص���ֵ����С��ֵ�ᵼ�¸�������ر����Ϊ��Ӱ��
		nmixtures��Ĭ��ֵ5������ʾ����ģ���и�˹�ɷֵ��������ϴ��ֵ���Ը��õ���Ӧ���ӵı�������Ҳ�����Ӽ�������*/
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
		//�������ͼ��image1���ɼ���ͼ��image2��
		
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
