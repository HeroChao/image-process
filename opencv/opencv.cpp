#include <opencv2\opencv.hpp>
#include <iostream>
#include <opencv2/core/utils/logger.hpp>
#include <string>
#include<Imageprocess.h>
using namespace cv;
using namespace std;



cv::Mat Imageprocess::imgPreprocess(cv::Mat& img, const int& opening_times,const int& closing_times,const bool& blur)
{
	cv::Mat imgGray, imgBlur, imgCanny, imgDial, imgThre;
	cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY); // 转灰度图
	if (blur == true) {
		for (int i = 0; i < 2; i++) {
			GaussianBlur(imgGray, imgBlur, cv::Size(5, 5), 1); // 高斯模糊
			boxFilter(imgBlur, imgBlur, imgBlur.type(), Size(5, 5));
			imgGray = imgBlur.clone();
		}
		
	}
	else {
		imgBlur = imgGray.clone();
	}
	imshow("imgBlur", imgBlur);
	// 高斯滤波
	Canny(imgBlur, imgCanny, 15, 100);
	cv::Mat kernel = getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)); // 创建了一个3*3大小的全1矩阵作为卷积核
	cv::Mat kernel2 = getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
	dilate(imgCanny, imgDial, kernel, cv::Point(-1, -1), opening_times);
	//dilate(imgDial, imgDial, kernel2, cv::Point(-1, -1), opening_times); // 膨胀
	erode(imgDial, imgThre, kernel, cv::Point(-1, -1), closing_times);    
	//erode(imgThre, imgThre, kernel2, cv::Point(-1, -1), closing_times);  // 腐蚀
	return imgThre;
}

cv::Mat Imageprocess::histogram_enhancement(cv::Mat& image)
{
	cv::Mat dst;
	int dims = image.channels();
	if (dims == 1) {
		equalizeHist(image, dst);
		imshow("图像均衡化1",dst);
	}
	else if (dims == 3) {
		vector<cv::Mat> channels;
		split(image, channels);
		equalizeHist(channels[0], channels[0]);
		merge(channels, dst);
		imshow("图像均衡化2", dst);
	}
	return dst;
}
 double Imageprocess::Image_rectangle(cv::Mat& image)//图像最小矩形
{
	//寻找最外层轮廓
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(image, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE, Point());
	cv::Mat imageContours = cv::Mat::zeros(image.size(), CV_8UC1);	//最小外接矩形画布

	vector<double> Contour_area;
	for (int i = 0; i < contours.size(); i++)
	{
		//绘制轮廓
		drawContours(imageContours, contours, i, Scalar(255), 1, 8, hierarchy);
		//Contour_area.push_back( contourArea(contours[i]));
		//绘制轮廓的最小外结矩形
		RotatedRect rect = minAreaRect(contours[i]);
		Rect rect1 = boundingRect(contours[i]);//最小正矩形；
		Point2f P[4];//初始化矩形四个顶点坐标
		rect.points(P);
		//for (int j = 0; j <= 3; j++)
		//{
			//line(imageContours, P[j], P[(j + 1) % 4], Scalar(255), 1);
		//}
		rectangle(imageContours, rect1, Scalar(255), 1, 8);
	}
	imshow("MinAreaRect", imageContours);

	//double Maximum_contour_area= *max_element(Contour_area.begin(), Contour_area.end());
	double Maximum_contour_area = 7;
	return Maximum_contour_area;
}

pair<cv::Mat,cv::Mat> Imageprocess::image_segmentation(cv::Mat& image)
{
	pair<cv::Mat, cv::Mat> Image;
	int Width = image.size().width;
	int Height = image.size().height;
	//cout<<Width<<image.cols<<Height<<image.rows<<endl;
	cv::Mat crop_L = image(Range(0, image.rows), Range(0, int(image.cols/2)));
	cv::Mat crop_R = image(Range(0, image.rows), Range(int(image.cols / 2), image.cols));
	//imshow("切割左图", crop_L);
	//imshow("切割右图", crop_R);
	Image = make_pair(crop_L, crop_R);
	return Image;
}

void VideoOperations(cv::Mat& frame)
{
	Imageprocess Mg;

	cv::Mat image1, image2;
	pair<cv::Mat, cv::Mat> Image = Mg.image_segmentation(frame);
	image1 = Image.first;
	image2 = Image.second;
	cv::cvtColor(image1, image1, COLOR_BGR2GRAY);
	cv::cvtColor(image2, image2, COLOR_BGR2GRAY);
	//Mg.Sift_detection(image1, image2);
	//Mg.Imageblur(frame);
	//Mg.feature_detection(image1, image2);
	//Mg.ORB_demo(500, 0, image1, image2);
}


cv::Mat I;//输入的图像矩阵
cv::Mat F;//图像的快速傅里叶变换
Point maxLoc;//傅里叶谱的最大值的坐标
int radius = 20;//截断频率
const int Max_RADIUS = 100;//设置最大的截断频率
cv::Mat lpFilter;//低通滤波器
int lpType = 0;//低通滤波器的类型
const int MAX_LPTYPE = 2;
cv::Mat F_lpFilter;//低通傅里叶变换
cv::Mat FlpSpectrum;//低通傅里叶变换的傅里叶谱的灰度级
cv::Mat result;//低通滤波后的效果
string lpFilterspectrum = "低通傅里叶谱";
//低通滤波器类型：理想低通滤波器、巴特沃斯低通滤波器、高斯低通滤波器
enum LPFILTER_TYPE { ILP_FILTER = 0, BLP_FILTER = 1, GLP_FILTER = 2 };
void fft2Image(cv::Mat I, cv::Mat& F)
{
	//得到I的行数和列数
	int rows = I.rows;
	int cols = I.cols;
	//满足快速傅里叶变换的最优行数和列数
	int rPadded = getOptimalDFTSize(rows);
	int cPadded = getOptimalDFTSize(cols);
	//左侧和下侧补0
	cv::Mat f;
	copyMakeBorder(I, f, 0, rPadded - rows, 0, cPadded - cols, BORDER_CONSTANT, Scalar::all(0));
	//单通道转为双通道
	cv::Mat planes[] = { Mat_<float>(f), cv::Mat::zeros(f.size(), CV_32F) };
	merge(planes, 1, f);
	cout << f.type() << endl;
	//快速傅里叶变换（双通道，用于存储实部和虚部）
	dft(f, F, DFT_COMPLEX_OUTPUT);
	cout << "Nihao" << endl;
}

void amplitudeSpectrum(InputArray _srcFFT, OutputArray _dstSpectrum)
{

	//判断傅里叶变换有两个通道
	CV_Assert(_srcFFT.channels() == 2);
	//分离通道
	vector<cv::Mat> FFT2Channel;
	split(_srcFFT, FFT2Channel);
	//计算傅里叶变换的幅度谱 sqrt(pow(R,2)+pow(I,2))
	magnitude(FFT2Channel[0], FFT2Channel[1], _dstSpectrum);
}

cv::Mat graySpectrum(cv::Mat spectrum)
{
	cv::Mat dst;
	log(spectrum + 1, dst);
	//归一化
	normalize(dst, dst, 0, 1, NORM_MINMAX);
	//为了进行灰度级显示，做类型转换
	dst.convertTo(dst, CV_32FC1, 255, 0);
	return dst;
}

cv::Mat createLPFilter(Size size, Point center, float radius, int type, int n = 2)
{
	cv::Mat lpFilter = cv::Mat::zeros(size, CV_32FC1);
	int rows = size.height;
	int cols = size.width;
	if (radius <= 0)
		return lpFilter;
	//构建理想低通滤波器
	if (type == ILP_FILTER)
	{
		for (int r = 0; r < rows; r++)
		{
			for (int c = 0; c < cols; c++)
			{
				float norm2 = pow(abs(float(r - center.y)), 2) + pow(abs(float(c - center.x)), 2);
				if (sqrt(norm2) < radius)
					lpFilter.at<float>(r, c) = 1;
				else
					lpFilter.at<float>(r, c) = 0;
			}
		}
	}
	//构建巴特沃斯低通滤波器
	if (type == BLP_FILTER)
	{
		for (int r = 0; r < rows; r++)
		{
			for (int c = 0; c < cols; c++)
			{
				lpFilter.at<float>(r, c) = float(1.0 / (1.0 + pow(sqrt(pow(r - center.y, 2.0) + pow(c - center.x, 2.0)) / radius, 2.0 * n)));
			}
		}
	}
	//构建高斯低通滤波器
	if (type == GLP_FILTER)
	{
		for (int r = 0; r < rows; r++)
		{
			for (int c = 0; c < cols; c++)
			{
				lpFilter.at<float>(r, c) = float(exp(-(pow(c - center.x, 2.0) + pow(r - center.y, 2.0)) / (2 * pow(radius, 2.0))));
			}
		}
	}
	return lpFilter;
}
//回调函数：调整低通滤波器的类型及截断频率
void callback_lpFilter(int, void*)
{
	/* -- 第五步：构建低通滤波器 -- */
	cv::Mat lpFilter = createLPFilter(F.size(), maxLoc, radius, lpType, 2);
	/*-- 第六步：低通滤波器和图像的快速傅里叶变换点乘 --*/
	F_lpFilter.create(F.size(), F.type());
	for (int r = 0; r < F_lpFilter.rows; r++)
	{
		for (int c = 0; c < F_lpFilter.cols; c++)
		{
			//分别取出当前位置的快速傅里叶变换和理想低通滤波器的值
			Vec2f F_rc = F.at<Vec2f>(r, c);
			float lpFilter_rc = lpFilter.at<float>(r, c);
			//低通滤波器和图像的快速傅里叶变换的对应位置相乘
			F_lpFilter.at<Vec2f>(r, c) = F_rc * lpFilter_rc;
		}
	}
	//低通傅里叶变换的傅里叶谱
	amplitudeSpectrum(F_lpFilter, FlpSpectrum);
	//低通傅里叶谱的灰度级显示
	FlpSpectrum = graySpectrum(FlpSpectrum);
	imshow(lpFilterspectrum, FlpSpectrum);
	/* -- 第七、八步：对低通傅里叶变换执行傅里叶逆变换，并只取实部 -- */
	cout << F_lpFilter.type() << endl;
	dft(F_lpFilter, result, DFT_SCALE + DFT_INVERSE + DFT_REAL_OUTPUT);
	/* -- 第九步：同乘以(-1)^(x+y) -- */
	for (int r = 0; r < result.rows; r++)
	{
		for (int c = 0; c < result.cols; c++)
		{
			if ((r + c) % 2)
				result.at<float>(r, c) *= -1;
		}
	}
	//注意将结果转换为 CV_8U 类型
	result.convertTo(result, CV_8UC1, 1.0, 0);
	/* -- 第十步：截取左上部分,其大小与输入图像的大小相同--*/
	result = result(Rect(0, 0, I.cols, I.rows)).clone();
	//imshow("经过低通滤波后的图片", result);
}


void Imageprocess::Imageblur(cv::Mat& image)
{
	cv::Mat src_image, dst_image;
	src_image = image.clone();
	//Laplacian(image, src_image, CV_32FC3, 3, 1, 0, BORDER_CONSTANT);

	cv::Mat fI;
	image.convertTo(fI, CV_32FC1, 1.0, 0.0);
	/* -- 第二步：每一个数乘以(-1)^(r+c) -- */
	for (int r = 0; r < fI.rows; r++)
	{
		for (int c = 0; c < fI.cols; c++)
		{
			if ((r + c) % 2)
			{
				fI.at<float>(r, c) *= -1;
			}
		}
	}
	cout << fI.type() << endl;
	/* -- 第三、四步：补0和快速傅里叶变换 -- */
	
	fft2Image(fI, F);
	cout << "Nihao" << endl;
	//傅里叶谱
	cv::Mat amplSpec;
	amplitudeSpectrum(F, amplSpec);
	//傅里叶谱的灰度级显示
	cv::Mat spectrum = graySpectrum(amplSpec);
	imshow("原傅里叶谱的灰度级显示", spectrum);
	imwrite("spectrum.jpg", spectrum);
	//找到傅里叶谱的最大值的坐标
	minMaxLoc(spectrum, NULL, NULL, NULL, &maxLoc);
	/* -- 低通滤波 -- */
	namedWindow(lpFilterspectrum, WINDOW_AUTOSIZE);
	createTrackbar("低通类型:", lpFilterspectrum, &lpType, MAX_LPTYPE,
		callback_lpFilter);
	createTrackbar("半径:", lpFilterspectrum, &radius, Max_RADIUS,
		callback_lpFilter);
	callback_lpFilter(0, 0);

}
void Imageprocess::Sift_detection(cv::Mat& image1, cv::Mat& image2)
{
	int numFeatures = 10000;
	//创建detector存放到KeyPoints中
	Ptr<SIFT> detector = SIFT::create(numFeatures);
	vector<KeyPoint> keypoints, keypoints2;
	detector->detect(image1, keypoints);
	detector->detect(image2, keypoints2);

	cv::Mat drawsrc, drawsrc2;
	drawKeypoints(image1, keypoints, drawsrc);
	drawKeypoints(image2, keypoints2, drawsrc2);

	//计算特征点描述符,特征向量提取
	cv::Mat dstSIFT, dstSIFT2;
	Ptr<SiftDescriptorExtractor> descriptor = SiftDescriptorExtractor::create();
	descriptor->compute(image1, keypoints, dstSIFT);
	descriptor->compute(image2, keypoints2, dstSIFT2);

	//进行BFMatch暴力匹配
	BFMatcher matcher(NORM_L2);
	//定义匹配结果变量
	vector<DMatch> matches;
	//实现描述符之间的匹配
	matcher.match(dstSIFT, dstSIFT2, matches);


	//定义向量距离的最大值与最小值
	double max_dist = 0;
	double min_dist = 1000;
	for (int i = 1; i < dstSIFT.rows; ++i)
	{
		//通过循环更新距离，距离越小越匹配
		double dist = matches[i].distance;
		if (dist > max_dist)
			max_dist = dist;
		if (dist < min_dist)
			min_dist = dist;
	}
	//匹配结果筛选    
	vector<DMatch> goodMatches;
	for (int i = 0; i < matches.size(); ++i)
	{
		double dist = matches[i].distance;
		if (dist < 2 * min_dist)
			goodMatches.push_back(matches[i]);
	}


	cv::Mat result;
	//匹配特征点天蓝色，单一特征点颜色随机
	drawMatches(image1, keypoints, image2, keypoints2, goodMatches, result,
		Scalar(255, 255, 0), Scalar::all(-1));
	imshow("Result", result);

	vector<Point2f> obj;
	vector<Point2f>scene;
	for (size_t i = 0; i < goodMatches.size(); i++)
	{
		obj.push_back(keypoints[goodMatches[i].queryIdx].pt);
		scene.push_back(keypoints2[goodMatches[i].trainIdx].pt);
	}
	vector<Point2f> obj_corner(4);
	vector<Point2f> scene_corner(4);
	//生成透视矩阵
	cv::Mat H = findHomography(obj, scene, RANSAC);

	obj_corner[0] = Point(0, 0);
	obj_corner[1] = Point(image1.cols, 0);
	obj_corner[2] = Point(image1.cols, image1.rows);
	obj_corner[3] = Point(0, image1.rows);
	//透视变换
	perspectiveTransform(obj_corner, scene_corner, H);
	cv::Mat resultImg = image2.clone();
	for (int i = 0; i < 4; i++)
	{
		line(resultImg, scene_corner[i] , scene_corner[(i + 1) % 4], Scalar(0, 0, 255), 2, 8, 0);
	}
	imshow("result image", resultImg);
}
void Imageprocess::feature_detection(cv::Mat& image1,cv::Mat& image2)
{
	Ptr<ORB> orb = cv::ORB::create(500);
	vector<KeyPoint> kp1, kp2;
	cv::Mat des1, des2;
	des1.convertTo(des1, CV_8UC1);
	des2.convertTo(des2, CV_8UC1);
	orb->detectAndCompute(image1, noArray(), kp1, des1);
	orb->detectAndCompute(image2, noArray(), kp2, des2);
	BFMatcher bf(NORM_HAMMING, true);//暴力匹配算法
	vector<DMatch> matches;
	bf.match(des1, des2, matches);
	sort(matches.begin(), matches.end(),[](const cv::DMatch& a, const DMatch& b) 
		{
			return a.distance < b.distance;
		});
	cv::Mat match_img;
	match_img.convertTo(match_img, CV_8UC1);
	drawMatches(image1, kp1, image2, kp2, matches,match_img, 
		Scalar::all(-1),Scalar::all(-1), vector<char>(),DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	imshow("Matches", match_img);
	
}
void Imageprocess::ORB_demo(int, void*,cv::Mat& img1,cv::Mat& img2)
{
	int Hession = 500;
	double t1 = getTickCount();
	//特征点提取
	Ptr<ORB> detector = ORB::create(500);
	vector<KeyPoint> keypoints_obj;
	vector<KeyPoint> keypoints_scene;
	//定义描述子
	cv::Mat descriptor_obj, descriptor_scene;
	//检测并计算成描述子
	
	detector->detectAndCompute(img1, cv::Mat(), keypoints_obj, descriptor_obj);
	detector->detectAndCompute(img2, cv::Mat(), keypoints_scene, descriptor_scene);

	double t2 = getTickCount();
	double t = (t2 - t1) * 1000 / getTickFrequency();
	//特征匹配
	FlannBasedMatcher fbmatcher(new flann::LshIndexParams(20, 10, 2));
	vector<DMatch> matches;
	//将找到的描述子进行匹配并存入matches中
	fbmatcher.match(descriptor_obj, descriptor_scene, matches);
	
	double minDist = 1000;
	double maxDist = 0;
	//找出最优描述子
	vector<DMatch> goodmatches;
	for (int i = 0; i < descriptor_obj.rows; i++)
	{
		double dist = matches[i].distance;
		if (dist < minDist)
		{
			minDist = dist;
		}
		if (dist > maxDist)
		{
			maxDist = dist;
		}

	}
	for (int i = 0; i < descriptor_obj.rows; i++)
	{
		double dist = matches[i].distance;
		if (dist < max(2 * minDist, 0.02))
		{
			goodmatches.push_back(matches[i]);
		}
	}
	cv::Mat orbImg;
	orbImg.convertTo(orbImg, CV_8UC3);
	drawMatches(img1, keypoints_obj, img2, keypoints_scene, goodmatches, orbImg,
		Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	//----------目标物体用矩形标识出来------------
	vector<Point2f> obj;
	vector<Point2f>scene;
	for (size_t i = 0; i < goodmatches.size(); i++)
	{
		obj.push_back(keypoints_obj[goodmatches[i].queryIdx].pt);
		scene.push_back(keypoints_scene[goodmatches[i].trainIdx].pt);
	}
	vector<Point2f> obj_corner(4);
	vector<Point2f> scene_corner(4);
	//生成透视矩阵
	cv::Mat H = findHomography(obj, scene, RANSAC);

	obj_corner[0] = Point(0, 0);
	obj_corner[1] = Point(img1.cols, 0);
	obj_corner[2] = Point(img1.cols, img1.rows);
	obj_corner[3] = Point(0, img1.rows);
	//透视变换
	perspectiveTransform(obj_corner, scene_corner, H);
	cv::Mat resultImg = orbImg.clone();


	for (int i = 0; i < 4; i++)
	{
		line(resultImg, scene_corner[i] + Point2f(img1.cols, 0), scene_corner[(i + 1) % 4] + Point2f(img1.cols, 0), Scalar(0, 0, 255), 2, 8, 0);
	}
	imshow("result image", resultImg);




	cout << "ORB执行时间为:" << t << "ms" << endl;
	cout << "最小距离为：" << minDist << endl;
	cout << "最大距离为：" << maxDist << endl;
	imshow("ORB_demo", orbImg);
}
cv::Mat Imageprocess::image_rendering(cv::Mat& image1, cv::Mat& image2)
{
	if (image1.size() != image2.size()) {
		cout << "图片大小不一致" << endl;
	}
	cv::Mat difference_image=cv::Mat::zeros(image1.size(),CV_32F);
	int dims = image1.channels();
	for (int i = 0; i < image1.rows; i++) {
		uchar* current_row = difference_image.ptr<uchar>(i);
		uchar* current_row1 = image1.ptr<uchar>(i);
		uchar* current_row2 = image2.ptr<uchar>(i);
		for (int j = 0; j < image1.cols; j++) {
			if (dims == 1) {

				int pv1 = *current_row1;
				int pv2 = *current_row2;
				*current_row++ = pv1 - pv2;
			}
			if (dims == 3) {
				*current_row++ = *current_row1 - *current_row2;
				*current_row++ = *current_row1 - *current_row2;
				*current_row++ = *current_row1 - *current_row2;
				
			}
		}
	}
		double minval,maxval;
		Point minloc;
		Point maxloc;
		imshow("图像差异", difference_image);
		minMaxLoc(difference_image, &minval, &maxval, &minloc, &maxloc);
		difference_image = 255.0 / (maxval - minval) * (difference_image - minval) ;
		
		return difference_image;
}

void Imageprocess::Image_changescore(cv::Mat& image1, cv::Mat& image2) 
{
	cv::Mat mean1, stddev1;
	meanStdDev(image1, mean1, stddev1);
	cv::Mat mean2, stddev2;
	meanStdDev(image2, mean2, stddev2);


	if ((abs(mean1.at<double>(0,0) - mean2.at<double>(0, 0)) < 10) && 
		(abs(stddev1.at<double>(0, 0) - stddev2.at<double>(0, 0)) > 2)) {

		cout << "产生气体" << endl;
	}
	else {
		cout << "无气体产生" << endl;
	}
}
cv::Mat Imageprocess::alignImages(cv::Mat& im1, cv::Mat& im2) {
	// 将图像转为灰度图
	cv::Mat im1Gray, im2Gray;
	cv::cvtColor(im1, im1Gray, COLOR_BGR2GRAY);
	cv::cvtColor(im2, im2Gray, COLOR_BGR2GRAY);
	// 检测ORB特征并计算描述符
	Ptr<ORB> orb = ORB::create(MAX_FEATURES);
	vector<KeyPoint> keypoints1, keypoints2;
	cv::Mat descriptors1, descriptors2;
	orb->detectAndCompute(im1Gray, noArray(), keypoints1, descriptors1);
	orb->detectAndCompute(im2Gray, noArray(), keypoints2, descriptors2);
	// 匹配特征点
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE_HAMMING);
	vector<DMatch> matches;
	matcher->match(descriptors1, descriptors2, matches, noArray());
	// 根据匹配得分对匹配点进行排序
	sort(matches.begin(), matches.end());
	// 移除较差的匹配点
	int numGoodMatches = static_cast<int>(matches.size() * GOOD_MATCH_PERCENT);
	matches.erase(matches.begin() + numGoodMatches, matches.end());
	// 绘制排名靠前的匹配点
	cv::Mat imMatches;
	drawMatches(im1, keypoints1, im2, keypoints2, matches, imMatches);
	// 提取好的匹配点位置
	vector<Point2f> points1, points2;
	for (int i = 0; i < numGoodMatches; ++i) {
		points1.push_back(keypoints1[matches[i].queryIdx].pt);
		points2.push_back(keypoints2[matches[i].trainIdx].pt);
	}
	// 计算单应性矩阵
	cv::Mat h = findHomography(points1, points2, RANSAC);
	// 将热红外图像使用单应性矩阵转换得到对齐后的结果
	Size size(im2.cols, im2.rows);
	cv::Mat im1Reg;
	warpPerspective(im1, im1Reg, h, size);
	imshow("im1Reg", im1Reg);
	return im1Reg;
}
cv::Mat Imageprocess::transform1(cv::Mat& image1,cv::Mat& image2) 
{
	cv::Mat templateImg = image1.clone();
	cv::Mat templateGray;
	cv::cvtColor(templateImg, templateGray, COLOR_BGR2GRAY);
	Canny(templateGray, templateGray, 50, 200);
	int tH = templateImg.rows, tW = templateImg.cols;
	imshow("Template", templateGray);

	// 循环处理所有需要找到模板的图像

	// 加载图像并进行灰度化，初始化用于跟踪匹配区域的变量
	cv::Mat image = image2.clone();
	cv::Mat gray;
	cv::cvtColor(image, gray, COLOR_BGR2GRAY);
	float foundVal = 0;
	Point maxLoc, tL, bR;

	// 循环处理图像的不同比例
	for (float scale = 1.0; scale >= 0.2; scale -= 0.05) {
		// 按比例调整图像大小并记录缩放比率
		cv::Mat resized;
		cv::resize(gray, resized, Size(), scale, scale);
		float r = static_cast<float>(gray.rows) / static_cast<float>(resized.rows);

		// 如果缩放后的图像小于模板，则退出循环
		if (resized.size().height < tH || resized.size().width < tW) {
			break;
		}

		// 在缩放后的灰度图像中检测边缘，并进行模板匹配以查找模板在图像中的位置
		cv::Mat edged;
		Canny(resized, edged, 50, 200);
		cv::Mat result;
		matchTemplate(edged, templateGray, result, TM_CCOEFF);

		double minVal, maxVal;
		Point minLoc, maxLoc;
		minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);

		// 如果需要可视化，则绘制一个框框显示匹配区域
		
			// 在检测到的区域周围画一个矩形
		cv::Mat clone;
		cv::cvtColor(edged, clone, COLOR_GRAY2BGR);
		cv::rectangle(clone, maxLoc, Point(maxLoc.x + tW, maxLoc.y + tH), Scalar(0, 0, 255), 2);
		imshow("Visualized", clone);

		// 如果得分比当前最大值更高，则更新变量
		if (foundVal < maxVal) {
			foundVal = maxVal;
			maxLoc = Point(maxLoc.x * scale, maxLoc.y * scale);
			tL = maxLoc;
			bR = Point(tL.x + tW, tL.y + tH);
		}
	}

	// 在检测到的区域周围画一个矩形并显示图像
	rectangle(image, tL, bR, Scalar(0, 0, 255), 2);
	imshow("Image", image);

	// 将匹配部分裁剪出来
	cv::Mat cropImg = image(Rect(tL, bR));
	imshow("Cropped Image", cropImg);

	// 拼接两幅图像，并保存在名为 results 的文件夹中
	cv::Mat thermalImg = image1.clone();
	resize(cropImg, cropImg, Size(thermalImg.cols, thermalImg.rows));
	cv::Mat final;
	hconcat(cropImg, thermalImg, final);
 
	// 读取参考图像
	
	cv::Mat imReference = image2.clone();

	// 读取需要对齐的图像
	cv::Mat im = image1.clone();
	//cv::Mat imReg = alignImages(im, imReference);

	return im;
}
cv::Mat Imageprocess::boundary_extraction(cv::Mat& image,const int& a,const int& b,bool c) {

	cv::Mat grayImage;
	cv::Mat srcImage1 = image.clone();
	cv::cvtColor(image, grayImage, COLOR_BGR2GRAY);
	cv::Mat dstImage, edge;

	blur(grayImage, grayImage, Size(3, 3));
	Canny(grayImage, edge, a, b, 3,c);

	dstImage.create(srcImage1.size(), srcImage1.type());
	dstImage = Scalar::all(0);
	srcImage1.copyTo(dstImage, edge);
	imshow("canny.jpg", dstImage);
	return edge;
}

Mat Imageprocess::image_contrast_enhancement(Mat image) {
	cout << image.channels() << image.size() << endl;
	Mat clahe_img;
	cvtColor(image, clahe_img, COLOR_BGR2Lab);
	vector<cv::Mat> channels(3);
	split(clahe_img, channels);

	Ptr<cv::CLAHE> clahe = createCLAHE();
	// 直方图的柱子高度大于计算后的ClipLimit的部分被裁剪掉，然后将其平均分配给整张直方图   
	clahe->setClipLimit(5.0); // (int)(4.*(8*8)/256)  
	clahe->setTilesGridSize(Size(8, 8)); // 将图像分为8*8块  
	Mat dst;
	clahe->apply(channels[0], dst);
	dst.copyTo(channels[0]);
	clahe->apply(channels[1], dst);
	dst.copyTo(channels[1]);
	clahe->apply(channels[2], dst);
	dst.copyTo(channels[2]);
	merge(channels, clahe_img);

	Mat image_clahe;
	cvtColor(clahe_img, image_clahe, COLOR_Lab2BGR);
	
	cvtColor(image_clahe, image_clahe, COLOR_BGR2GRAY);
	cout << image_clahe.channels()<< image_clahe.size() << endl;
	imshow("CLAHE Image", image_clahe);
	return image_clahe;
}