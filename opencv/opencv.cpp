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
	cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY); // ת�Ҷ�ͼ
	if (blur == true) {
		for (int i = 0; i < 2; i++) {
			GaussianBlur(imgGray, imgBlur, cv::Size(5, 5), 1); // ��˹ģ��
			boxFilter(imgBlur, imgBlur, imgBlur.type(), Size(5, 5));
			imgGray = imgBlur.clone();
		}
		
	}
	else {
		imgBlur = imgGray.clone();
	}
	imshow("imgBlur", imgBlur);
	// ��˹�˲�
	Canny(imgBlur, imgCanny, 15, 100);
	cv::Mat kernel = getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)); // ������һ��3*3��С��ȫ1������Ϊ�����
	cv::Mat kernel2 = getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
	dilate(imgCanny, imgDial, kernel, cv::Point(-1, -1), opening_times);
	//dilate(imgDial, imgDial, kernel2, cv::Point(-1, -1), opening_times); // ����
	erode(imgDial, imgThre, kernel, cv::Point(-1, -1), closing_times);    
	//erode(imgThre, imgThre, kernel2, cv::Point(-1, -1), closing_times);  // ��ʴ
	return imgThre;
}

cv::Mat Imageprocess::histogram_enhancement(cv::Mat& image)
{
	cv::Mat dst;
	int dims = image.channels();
	if (dims == 1) {
		equalizeHist(image, dst);
		imshow("ͼ����⻯1",dst);
	}
	else if (dims == 3) {
		vector<cv::Mat> channels;
		split(image, channels);
		equalizeHist(channels[0], channels[0]);
		merge(channels, dst);
		imshow("ͼ����⻯2", dst);
	}
	return dst;
}
 double Imageprocess::Image_rectangle(cv::Mat& image)//ͼ����С����
{
	//Ѱ�����������
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(image, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE, Point());
	cv::Mat imageContours = cv::Mat::zeros(image.size(), CV_8UC1);	//��С��Ӿ��λ���

	vector<double> Contour_area;
	for (int i = 0; i < contours.size(); i++)
	{
		//��������
		drawContours(imageContours, contours, i, Scalar(255), 1, 8, hierarchy);
		//Contour_area.push_back( contourArea(contours[i]));
		//������������С������
		RotatedRect rect = minAreaRect(contours[i]);
		Rect rect1 = boundingRect(contours[i]);//��С�����Σ�
		Point2f P[4];//��ʼ�������ĸ���������
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
	//imshow("�и���ͼ", crop_L);
	//imshow("�и���ͼ", crop_R);
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


cv::Mat I;//�����ͼ�����
cv::Mat F;//ͼ��Ŀ��ٸ���Ҷ�任
Point maxLoc;//����Ҷ�׵����ֵ������
int radius = 20;//�ض�Ƶ��
const int Max_RADIUS = 100;//�������Ľض�Ƶ��
cv::Mat lpFilter;//��ͨ�˲���
int lpType = 0;//��ͨ�˲���������
const int MAX_LPTYPE = 2;
cv::Mat F_lpFilter;//��ͨ����Ҷ�任
cv::Mat FlpSpectrum;//��ͨ����Ҷ�任�ĸ���Ҷ�׵ĻҶȼ�
cv::Mat result;//��ͨ�˲����Ч��
string lpFilterspectrum = "��ͨ����Ҷ��";
//��ͨ�˲������ͣ������ͨ�˲�����������˹��ͨ�˲�������˹��ͨ�˲���
enum LPFILTER_TYPE { ILP_FILTER = 0, BLP_FILTER = 1, GLP_FILTER = 2 };
void fft2Image(cv::Mat I, cv::Mat& F)
{
	//�õ�I������������
	int rows = I.rows;
	int cols = I.cols;
	//������ٸ���Ҷ�任����������������
	int rPadded = getOptimalDFTSize(rows);
	int cPadded = getOptimalDFTSize(cols);
	//�����²ಹ0
	cv::Mat f;
	copyMakeBorder(I, f, 0, rPadded - rows, 0, cPadded - cols, BORDER_CONSTANT, Scalar::all(0));
	//��ͨ��תΪ˫ͨ��
	cv::Mat planes[] = { Mat_<float>(f), cv::Mat::zeros(f.size(), CV_32F) };
	merge(planes, 1, f);
	cout << f.type() << endl;
	//���ٸ���Ҷ�任��˫ͨ�������ڴ洢ʵ�����鲿��
	dft(f, F, DFT_COMPLEX_OUTPUT);
	cout << "Nihao" << endl;
}

void amplitudeSpectrum(InputArray _srcFFT, OutputArray _dstSpectrum)
{

	//�жϸ���Ҷ�任������ͨ��
	CV_Assert(_srcFFT.channels() == 2);
	//����ͨ��
	vector<cv::Mat> FFT2Channel;
	split(_srcFFT, FFT2Channel);
	//���㸵��Ҷ�任�ķ����� sqrt(pow(R,2)+pow(I,2))
	magnitude(FFT2Channel[0], FFT2Channel[1], _dstSpectrum);
}

cv::Mat graySpectrum(cv::Mat spectrum)
{
	cv::Mat dst;
	log(spectrum + 1, dst);
	//��һ��
	normalize(dst, dst, 0, 1, NORM_MINMAX);
	//Ϊ�˽��лҶȼ���ʾ��������ת��
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
	//���������ͨ�˲���
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
	//����������˹��ͨ�˲���
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
	//������˹��ͨ�˲���
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
//�ص�������������ͨ�˲��������ͼ��ض�Ƶ��
void callback_lpFilter(int, void*)
{
	/* -- ���岽��������ͨ�˲��� -- */
	cv::Mat lpFilter = createLPFilter(F.size(), maxLoc, radius, lpType, 2);
	/*-- ����������ͨ�˲�����ͼ��Ŀ��ٸ���Ҷ�任��� --*/
	F_lpFilter.create(F.size(), F.type());
	for (int r = 0; r < F_lpFilter.rows; r++)
	{
		for (int c = 0; c < F_lpFilter.cols; c++)
		{
			//�ֱ�ȡ����ǰλ�õĿ��ٸ���Ҷ�任�������ͨ�˲�����ֵ
			Vec2f F_rc = F.at<Vec2f>(r, c);
			float lpFilter_rc = lpFilter.at<float>(r, c);
			//��ͨ�˲�����ͼ��Ŀ��ٸ���Ҷ�任�Ķ�Ӧλ�����
			F_lpFilter.at<Vec2f>(r, c) = F_rc * lpFilter_rc;
		}
	}
	//��ͨ����Ҷ�任�ĸ���Ҷ��
	amplitudeSpectrum(F_lpFilter, FlpSpectrum);
	//��ͨ����Ҷ�׵ĻҶȼ���ʾ
	FlpSpectrum = graySpectrum(FlpSpectrum);
	imshow(lpFilterspectrum, FlpSpectrum);
	/* -- ���ߡ��˲����Ե�ͨ����Ҷ�任ִ�и���Ҷ��任����ֻȡʵ�� -- */
	cout << F_lpFilter.type() << endl;
	dft(F_lpFilter, result, DFT_SCALE + DFT_INVERSE + DFT_REAL_OUTPUT);
	/* -- �ھŲ���ͬ����(-1)^(x+y) -- */
	for (int r = 0; r < result.rows; r++)
	{
		for (int c = 0; c < result.cols; c++)
		{
			if ((r + c) % 2)
				result.at<float>(r, c) *= -1;
		}
	}
	//ע�⽫���ת��Ϊ CV_8U ����
	result.convertTo(result, CV_8UC1, 1.0, 0);
	/* -- ��ʮ������ȡ���ϲ���,���С������ͼ��Ĵ�С��ͬ--*/
	result = result(Rect(0, 0, I.cols, I.rows)).clone();
	//imshow("������ͨ�˲����ͼƬ", result);
}


void Imageprocess::Imageblur(cv::Mat& image)
{
	cv::Mat src_image, dst_image;
	src_image = image.clone();
	//Laplacian(image, src_image, CV_32FC3, 3, 1, 0, BORDER_CONSTANT);

	cv::Mat fI;
	image.convertTo(fI, CV_32FC1, 1.0, 0.0);
	/* -- �ڶ�����ÿһ��������(-1)^(r+c) -- */
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
	/* -- �������Ĳ�����0�Ϳ��ٸ���Ҷ�任 -- */
	
	fft2Image(fI, F);
	cout << "Nihao" << endl;
	//����Ҷ��
	cv::Mat amplSpec;
	amplitudeSpectrum(F, amplSpec);
	//����Ҷ�׵ĻҶȼ���ʾ
	cv::Mat spectrum = graySpectrum(amplSpec);
	imshow("ԭ����Ҷ�׵ĻҶȼ���ʾ", spectrum);
	imwrite("spectrum.jpg", spectrum);
	//�ҵ�����Ҷ�׵����ֵ������
	minMaxLoc(spectrum, NULL, NULL, NULL, &maxLoc);
	/* -- ��ͨ�˲� -- */
	namedWindow(lpFilterspectrum, WINDOW_AUTOSIZE);
	createTrackbar("��ͨ����:", lpFilterspectrum, &lpType, MAX_LPTYPE,
		callback_lpFilter);
	createTrackbar("�뾶:", lpFilterspectrum, &radius, Max_RADIUS,
		callback_lpFilter);
	callback_lpFilter(0, 0);

}
void Imageprocess::Sift_detection(cv::Mat& image1, cv::Mat& image2)
{
	int numFeatures = 10000;
	//����detector��ŵ�KeyPoints��
	Ptr<SIFT> detector = SIFT::create(numFeatures);
	vector<KeyPoint> keypoints, keypoints2;
	detector->detect(image1, keypoints);
	detector->detect(image2, keypoints2);

	cv::Mat drawsrc, drawsrc2;
	drawKeypoints(image1, keypoints, drawsrc);
	drawKeypoints(image2, keypoints2, drawsrc2);

	//����������������,����������ȡ
	cv::Mat dstSIFT, dstSIFT2;
	Ptr<SiftDescriptorExtractor> descriptor = SiftDescriptorExtractor::create();
	descriptor->compute(image1, keypoints, dstSIFT);
	descriptor->compute(image2, keypoints2, dstSIFT2);

	//����BFMatch����ƥ��
	BFMatcher matcher(NORM_L2);
	//����ƥ��������
	vector<DMatch> matches;
	//ʵ��������֮���ƥ��
	matcher.match(dstSIFT, dstSIFT2, matches);


	//����������������ֵ����Сֵ
	double max_dist = 0;
	double min_dist = 1000;
	for (int i = 1; i < dstSIFT.rows; ++i)
	{
		//ͨ��ѭ�����¾��룬����ԽСԽƥ��
		double dist = matches[i].distance;
		if (dist > max_dist)
			max_dist = dist;
		if (dist < min_dist)
			min_dist = dist;
	}
	//ƥ����ɸѡ    
	vector<DMatch> goodMatches;
	for (int i = 0; i < matches.size(); ++i)
	{
		double dist = matches[i].distance;
		if (dist < 2 * min_dist)
			goodMatches.push_back(matches[i]);
	}


	cv::Mat result;
	//ƥ������������ɫ����һ��������ɫ���
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
	//����͸�Ӿ���
	cv::Mat H = findHomography(obj, scene, RANSAC);

	obj_corner[0] = Point(0, 0);
	obj_corner[1] = Point(image1.cols, 0);
	obj_corner[2] = Point(image1.cols, image1.rows);
	obj_corner[3] = Point(0, image1.rows);
	//͸�ӱ任
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
	BFMatcher bf(NORM_HAMMING, true);//����ƥ���㷨
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
	//��������ȡ
	Ptr<ORB> detector = ORB::create(500);
	vector<KeyPoint> keypoints_obj;
	vector<KeyPoint> keypoints_scene;
	//����������
	cv::Mat descriptor_obj, descriptor_scene;
	//��Ⲣ�����������
	
	detector->detectAndCompute(img1, cv::Mat(), keypoints_obj, descriptor_obj);
	detector->detectAndCompute(img2, cv::Mat(), keypoints_scene, descriptor_scene);

	double t2 = getTickCount();
	double t = (t2 - t1) * 1000 / getTickFrequency();
	//����ƥ��
	FlannBasedMatcher fbmatcher(new flann::LshIndexParams(20, 10, 2));
	vector<DMatch> matches;
	//���ҵ��������ӽ���ƥ�䲢����matches��
	fbmatcher.match(descriptor_obj, descriptor_scene, matches);
	
	double minDist = 1000;
	double maxDist = 0;
	//�ҳ�����������
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
	//----------Ŀ�������þ��α�ʶ����------------
	vector<Point2f> obj;
	vector<Point2f>scene;
	for (size_t i = 0; i < goodmatches.size(); i++)
	{
		obj.push_back(keypoints_obj[goodmatches[i].queryIdx].pt);
		scene.push_back(keypoints_scene[goodmatches[i].trainIdx].pt);
	}
	vector<Point2f> obj_corner(4);
	vector<Point2f> scene_corner(4);
	//����͸�Ӿ���
	cv::Mat H = findHomography(obj, scene, RANSAC);

	obj_corner[0] = Point(0, 0);
	obj_corner[1] = Point(img1.cols, 0);
	obj_corner[2] = Point(img1.cols, img1.rows);
	obj_corner[3] = Point(0, img1.rows);
	//͸�ӱ任
	perspectiveTransform(obj_corner, scene_corner, H);
	cv::Mat resultImg = orbImg.clone();


	for (int i = 0; i < 4; i++)
	{
		line(resultImg, scene_corner[i] + Point2f(img1.cols, 0), scene_corner[(i + 1) % 4] + Point2f(img1.cols, 0), Scalar(0, 0, 255), 2, 8, 0);
	}
	imshow("result image", resultImg);




	cout << "ORBִ��ʱ��Ϊ:" << t << "ms" << endl;
	cout << "��С����Ϊ��" << minDist << endl;
	cout << "������Ϊ��" << maxDist << endl;
	imshow("ORB_demo", orbImg);
}
cv::Mat Imageprocess::image_rendering(cv::Mat& image1, cv::Mat& image2)
{
	if (image1.size() != image2.size()) {
		cout << "ͼƬ��С��һ��" << endl;
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
		imshow("ͼ�����", difference_image);
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

		cout << "��������" << endl;
	}
	else {
		cout << "���������" << endl;
	}
}
cv::Mat Imageprocess::alignImages(cv::Mat& im1, cv::Mat& im2) {
	// ��ͼ��תΪ�Ҷ�ͼ
	cv::Mat im1Gray, im2Gray;
	cv::cvtColor(im1, im1Gray, COLOR_BGR2GRAY);
	cv::cvtColor(im2, im2Gray, COLOR_BGR2GRAY);
	// ���ORB����������������
	Ptr<ORB> orb = ORB::create(MAX_FEATURES);
	vector<KeyPoint> keypoints1, keypoints2;
	cv::Mat descriptors1, descriptors2;
	orb->detectAndCompute(im1Gray, noArray(), keypoints1, descriptors1);
	orb->detectAndCompute(im2Gray, noArray(), keypoints2, descriptors2);
	// ƥ��������
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE_HAMMING);
	vector<DMatch> matches;
	matcher->match(descriptors1, descriptors2, matches, noArray());
	// ����ƥ��÷ֶ�ƥ����������
	sort(matches.begin(), matches.end());
	// �Ƴ��ϲ��ƥ���
	int numGoodMatches = static_cast<int>(matches.size() * GOOD_MATCH_PERCENT);
	matches.erase(matches.begin() + numGoodMatches, matches.end());
	// ����������ǰ��ƥ���
	cv::Mat imMatches;
	drawMatches(im1, keypoints1, im2, keypoints2, matches, imMatches);
	// ��ȡ�õ�ƥ���λ��
	vector<Point2f> points1, points2;
	for (int i = 0; i < numGoodMatches; ++i) {
		points1.push_back(keypoints1[matches[i].queryIdx].pt);
		points2.push_back(keypoints2[matches[i].trainIdx].pt);
	}
	// ���㵥Ӧ�Ծ���
	cv::Mat h = findHomography(points1, points2, RANSAC);
	// ���Ⱥ���ͼ��ʹ�õ�Ӧ�Ծ���ת���õ������Ľ��
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

	// ѭ������������Ҫ�ҵ�ģ���ͼ��

	// ����ͼ�񲢽��лҶȻ�����ʼ�����ڸ���ƥ������ı���
	cv::Mat image = image2.clone();
	cv::Mat gray;
	cv::cvtColor(image, gray, COLOR_BGR2GRAY);
	float foundVal = 0;
	Point maxLoc, tL, bR;

	// ѭ������ͼ��Ĳ�ͬ����
	for (float scale = 1.0; scale >= 0.2; scale -= 0.05) {
		// ����������ͼ���С����¼���ű���
		cv::Mat resized;
		cv::resize(gray, resized, Size(), scale, scale);
		float r = static_cast<float>(gray.rows) / static_cast<float>(resized.rows);

		// ������ź��ͼ��С��ģ�壬���˳�ѭ��
		if (resized.size().height < tH || resized.size().width < tW) {
			break;
		}

		// �����ź�ĻҶ�ͼ���м���Ե��������ģ��ƥ���Բ���ģ����ͼ���е�λ��
		cv::Mat edged;
		Canny(resized, edged, 50, 200);
		cv::Mat result;
		matchTemplate(edged, templateGray, result, TM_CCOEFF);

		double minVal, maxVal;
		Point minLoc, maxLoc;
		minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);

		// �����Ҫ���ӻ��������һ�������ʾƥ������
		
			// �ڼ�⵽��������Χ��һ������
		cv::Mat clone;
		cv::cvtColor(edged, clone, COLOR_GRAY2BGR);
		cv::rectangle(clone, maxLoc, Point(maxLoc.x + tW, maxLoc.y + tH), Scalar(0, 0, 255), 2);
		imshow("Visualized", clone);

		// ����÷ֱȵ�ǰ���ֵ���ߣ�����±���
		if (foundVal < maxVal) {
			foundVal = maxVal;
			maxLoc = Point(maxLoc.x * scale, maxLoc.y * scale);
			tL = maxLoc;
			bR = Point(tL.x + tW, tL.y + tH);
		}
	}

	// �ڼ�⵽��������Χ��һ�����β���ʾͼ��
	rectangle(image, tL, bR, Scalar(0, 0, 255), 2);
	imshow("Image", image);

	// ��ƥ�䲿�ֲü�����
	cv::Mat cropImg = image(Rect(tL, bR));
	imshow("Cropped Image", cropImg);

	// ƴ������ͼ�񣬲���������Ϊ results ���ļ�����
	cv::Mat thermalImg = image1.clone();
	resize(cropImg, cropImg, Size(thermalImg.cols, thermalImg.rows));
	cv::Mat final;
	hconcat(cropImg, thermalImg, final);
 
	// ��ȡ�ο�ͼ��
	
	cv::Mat imReference = image2.clone();

	// ��ȡ��Ҫ�����ͼ��
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
	// ֱ��ͼ�����Ӹ߶ȴ��ڼ�����ClipLimit�Ĳ��ֱ��ü�����Ȼ����ƽ�����������ֱ��ͼ   
	clahe->setClipLimit(5.0); // (int)(4.*(8*8)/256)  
	clahe->setTilesGridSize(Size(8, 8)); // ��ͼ���Ϊ8*8��  
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