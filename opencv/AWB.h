#include <opencv2\opencv.hpp>
#include <iostream>
#include <opencv2/core/utils/logger.hpp>
#include <string>

class AWB
{
public:
	 
cv::Mat white_balance_3(cv::Mat& img) {
        /*
            灰度世界假设
            @param img: 读取的图片数据，类型为 cv::Mat
            @return: 返回白平衡后的图片数据，类型为 cv::Mat。
        */
        cv::Mat dst_img = img.clone();
        std::vector<cv::Mat> bgr_channels;
        cv::split(dst_img, bgr_channels);

        double B_ave = cv::mean(bgr_channels[0])[0];
        double G_ave = cv::mean(bgr_channels[1])[0];
        double R_ave = cv::mean(bgr_channels[2])[0];
        double K = (B_ave + G_ave + R_ave) / 3.0;
        double Kb = K / B_ave, Kg = K / G_ave, Kr = K / R_ave;

        bgr_channels[0] = cv::min(bgr_channels[0] * Kb, 255.0);
        bgr_channels[1] = cv::min(bgr_channels[1] * Kg, 255.0);
        bgr_channels[2] = cv::min(bgr_channels[2] * Kr, 255.0);

        cv::merge(bgr_channels, dst_img);

        return dst_img;
}


private:

};
