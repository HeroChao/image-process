#include <opencv2\opencv.hpp>
#include <iostream>
#include <opencv2/core/utils/logger.hpp>
#include <string>


#define MAX_FEATURES 500 // �����������
#define GOOD_MATCH_PERCENT 0.15 // ����ɸѡƥ���ı���

class Imageprocess
{
public:


	cv::Mat imgPreprocess(cv::Mat& img,const int& opening_times,const int& closing_times,const bool& blur);//ͼ��Ԥ����
	cv::Mat histogram_enhancement(cv::Mat& image);//ֱ��ͼ��ǿ
	bool OpenCameraAndRecord(const std::string& url, cv::Mat& image);//��ʾ��Ƶ����;
	void Imageblur(cv::Mat& image);//ͼ���˲�����
	double Image_rectangle(cv::Mat& image);//ͼ����С����
	std::pair<cv::Mat, cv::Mat>  image_segmentation(cv::Mat& image);//�и�ͼ��
	void feature_detection(cv::Mat& image1, cv::Mat& image2);//bfͼ��ƥ��
	void ORB_demo(int, void*, cv::Mat& img1, cv::Mat& img2);//ORBƥ��
	cv::Mat image_rendering(cv::Mat& image1, cv::Mat& image2);//ͼ����Ⱦ
	void Image_changescore(cv::Mat& image1, cv::Mat& image2);//ͼ��Աȵ÷�
	void Sift_detection(cv::Mat& image1, cv::Mat& image2);//SIFT�㷨ƥ��
	cv::Mat alignImages(cv::Mat& im1, cv::Mat& im2);//����ƥ���㷨
	cv::Mat transform1(cv::Mat& image1, cv::Mat& image2);//���ƥ��ͼ��

	void gas_error_elimination(cv::Mat& src_image,cv::Mat& IRImage, cv::Mat& mask_image);//ȡ��ͼ�������������������
	void  BackGroundKnn(cv::VideoCapture& capture, cv::Ptr<cv::BackgroundSubtractorKNN> pBackgroundKnn);//knnͼ��ָ���
	cv::Mat boundary_extraction(cv::Mat& image,const int& a, const int& b,bool c);//ͼ��߽���ȡ��
	cv::Mat RegionGrowSegment(cv::Mat& srcImage, cv::Mat& MaskImage,const int& ch1Thres, const int& ch1LowerBind, const int& ch1UpperBind, const int& ch2Thres);//����������

	void VideoOperations(cv::Mat& frame);
	cv::Mat image_contrast_enhancement(cv::Mat image);//��ǿͼ��Աȶ�
	cv::Mat TraversePixels(cv::Mat image);//���ٱ������ص㣻
	bool RenderingJudgment(cv::Mat image, int renderthre);//�ж��Ƿ���Ⱦ��

private:


};
class CameraAlgorithm
{
public:
	void Autofocus(cv::Mat& image);//�Զ��Խ��㷨
	cv::Mat Image_color_enhancement(cv::Mat& image);//ͼ���ɫ��ǿ
	cv::Mat Image_selection(cv::Mat& image1, cv::Mat& image2);//ͼ������ѡȡ��
	cv::Mat ConnectNearRegion(cv::Mat& src);
	cv::Mat color_rendering(cv::Mat& image);
	cv::Mat boxfilter(cv::Mat& src,cv::Mat& mask);
	bool image_zero(cv::Mat image1, cv::Mat image2);
	cv::Mat GetMask(cv::Mat image);
	void logo_display(cv::Mat image, const int flag);
};

