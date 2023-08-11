#include <opencv2\opencv.hpp>
#include <iostream>
#include <opencv2/core/utils/logger.hpp>
#include <string>


#define MAX_FEATURES 500 // 最大特征点数
#define GOOD_MATCH_PERCENT 0.15 // 用于筛选匹配点的比例

class Imageprocess
{
public:


	cv::Mat imgPreprocess(cv::Mat& img,const int& opening_times,const int& closing_times,const bool& blur);//图像预处理
	cv::Mat histogram_enhancement(cv::Mat& image);//直方图增强
	bool OpenCameraAndRecord(const std::string& url, cv::Mat& image);//显示视频数据;
	void Imageblur(cv::Mat& image);//图像滤波处理
	double Image_rectangle(cv::Mat& image);//图像最小矩形
	std::pair<cv::Mat, cv::Mat>  image_segmentation(cv::Mat& image);//切割图像；
	void feature_detection(cv::Mat& image1, cv::Mat& image2);//bf图像匹配
	void ORB_demo(int, void*, cv::Mat& img1, cv::Mat& img2);//ORB匹配
	cv::Mat image_rendering(cv::Mat& image1, cv::Mat& image2);//图像渲染
	void Image_changescore(cv::Mat& image1, cv::Mat& image2);//图像对比得分
	void Sift_detection(cv::Mat& image1, cv::Mat& image2);//SIFT算法匹配
	cv::Mat alignImages(cv::Mat& im1, cv::Mat& im2);//红外匹配算法
	cv::Mat transform1(cv::Mat& image1, cv::Mat& image2);//随机匹配图像

	void gas_error_elimination(cv::Mat& src_image,cv::Mat& IRImage, cv::Mat& mask_image);//取得图像最大轮廓以消除噪声
	void  BackGroundKnn(cv::VideoCapture& capture, cv::Ptr<cv::BackgroundSubtractorKNN> pBackgroundKnn);//knn图像分割器
	cv::Mat boundary_extraction(cv::Mat& image,const int& a, const int& b,bool c);//图像边界提取；
	cv::Mat RegionGrowSegment(cv::Mat& srcImage, cv::Mat& MaskImage,const int& ch1Thres, const int& ch1LowerBind, const int& ch1UpperBind, const int& ch2Thres);//区域生长；

	void VideoOperations(cv::Mat& frame);
	cv::Mat image_contrast_enhancement(cv::Mat image);//增强图像对比度
	cv::Mat TraversePixels(cv::Mat image);//快速遍历像素点；
	bool RenderingJudgment(cv::Mat image, int renderthre);//判断是否渲染；

private:


};
class CameraAlgorithm
{
public:
	void Autofocus(cv::Mat& image);//自动对焦算法
	cv::Mat Image_color_enhancement(cv::Mat& image);//图像彩色增强
	cv::Mat Image_selection(cv::Mat& image1, cv::Mat& image2);//图像区域选取；
	cv::Mat ConnectNearRegion(cv::Mat& src);
	cv::Mat color_rendering(cv::Mat& image);
	cv::Mat boxfilter(cv::Mat& src,cv::Mat& mask);
	bool image_zero(cv::Mat image1, cv::Mat image2);
	cv::Mat GetMask(cv::Mat image);
	void logo_display(cv::Mat image, const int flag);
};

