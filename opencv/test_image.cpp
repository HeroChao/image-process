#include<fstream>
#include <string>
#include<Imageprocess.h>
#include <ctime>
#include<AWB.h>
#include<IR_VIS.h>
#include<FFmpeg.h>
#include <thread>
#include <mutex>
#include<future>




using namespace cv;
using namespace std;



void   Delay(int   time)//time*1000为秒数 
{
	clock_t   now = clock();

	while (clock() - now < time);
}
template<typename T>
std::map<std::string, T> Parameter = {
		{"DistanceThreshold",20},
		{"NumberOfAdoptions",0},
		{"IsContrastEnhanced", 0},
		{"BOXFIL",0},
		{"ThresholdPixel",310000},
		{"LowAlarmThreshold",100.0},
		{"HighAlarmThreshold",200000.0},
		{"ContinuousRenderingFrames",1}
};
int set_Parameter_input(const string path) {
	ifstream fin(path);
	if (!fin) {
		cerr << "Failed to open Parameter file: " << path << endl;
		return -1;
	}
	int count = 0;
	
	cout << Parameter<int>["LowAlarmThreshold"] << "OK" << endl;
	string line;
	while (getline(fin, line)) {
		for (auto& parameter : Parameter<int>) {
			if (line.rfind(parameter.first, 0) == 0) {
				parameter.second = stoi(line.substr(parameter.first.length()));
				cout << parameter.second << endl;
				count++;
				break;
			}
		}
	}
	cout << Parameter<int>["LowAlarmThreshold"] << "OK" << endl;
	return count;
}

int tcp_storage_command() {
	std::vector<int> numbers = { 7, 3, 3, 4, 5 }; // 数字数组

	std::string filename = "numbers.txt";
	std::ofstream file(filename); // 创建文件

	if (file.is_open()) { // 检查是否成功创建文件
		// 将数组中的数字逐个写入文件，每个数字占一行并以制表符分隔
		for (const auto& num : numbers) {
			file << num << "\t" ;
		}

		file.close(); // 关闭文件
		std::cout << "Numbers saved to file: " << filename << std::endl;
	}
	else {
		std::cout << "Failed to create file: " << filename << std::endl;
	}

	return 0;
}

int tcp_command_read(const string path) {
	ifstream file(path);
	if (!file.is_open()) {
		cerr << "Failed to open file: " << path << endl;
		return -1;
	} std::string line;
	if (std::getline(file, line)) {
		std::istringstream iss(line);
		std::vector<int> numbers;
		int number;
		while (iss >> number) {
			numbers.push_back(number);
		}
		// 输出读取的数字
		for (int num : numbers) {
			std::cout << num << " ";
		}
		std::cout << std::endl;
	}

	file.close();
	return 0;
}

std::vector<int> extractNumbers(const std::string& inputString) {
	std::vector<int> numbers;
	std::istringstream iss(inputString);
	std::string segment;

	while (std::getline(iss, segment, '_')) {
		try {
			int number = std::stoi(segment);
			numbers.push_back(number);
		}
		catch (const std::exception& e) {
			// 忽略非数字的分段
		}
	}

	return numbers;
}
int main(int argc, char** argv)
{
	utils::logging::setLogLevel(utils::logging::LOG_LEVEL_SILENT);

	cv::Mat image = imread("D:/文件安装包/1.png", IMREAD_ANYDEPTH | IMREAD_ANYCOLOR);
	cv::Mat image1 = imread("D:/文件安装包/2.png");
	
	
	if (image.empty())
	{
		cout << "找不到图像" << endl;
		return -1;
	}
	std::string inputString = "_111_0_0_10_0_1_1_22";
	std::vector<int> result = extractNumbers(inputString);

	for (int number : result) {
		std::cout << number << " ";
	}
	std::cout << std::endl;

	//namedWindow("输入窗口", WINDOW_AUTOSIZE);
	//imshow("图像显示1", image);
	//imshow("图像显示2", image1);
	Imageprocess Mg;
	CameraAlgorithm Mc;
	GasDetect gas;
	IR_VIS iv;
	GasProcess gp;
	//tcp_command_read("numbers.txt");
	//ffmpeg_encodec("rtsp://172.20.20.93:8554/live");
	//ffmpeg_decodec("rtsp://admin:zhgx1234@172.20.20.115:554/cam/realmonitor?channel=1&subtype=0");
	cv::waitKey(0);
	destroyAllWindows();
	return 0;

}

//int main(int argc, char* argv[])
//{
//	AVFormatContext* input_ctx = NULL;
//	int video_stream, ret;
//	AVStream* video = NULL;
//	AVCodecContext* decoder_ctx = NULL;
//	const AVCodec* decoder = NULL;
//	AVPacket packet;
//	enum AVHWDeviceType type;
//	int i;
//
//	if (argc < 4) {
//		fprintf(stderr, "Usage: %s <device type> <input file> <output file>\n", argv[0]);
//		return -1;
//	}
//	// 设备类型为：cuda dxva2 qsv d3d11va opencl，通常在windows使用d3d11va或者dxva2
//	type = av_hwdevice_find_type_by_name(argv[1]); //根据设备名找到设备类型
//	if (type == AV_HWDEVICE_TYPE_NONE) {
//		fprintf(stderr, "Device type %s is not supported.\n", argv[1]);
//		fprintf(stderr, "Available device types:");
//		while ((type = av_hwdevice_iterate_types(type)) != AV_HWDEVICE_TYPE_NONE)
//			fprintf(stderr, " %s", av_hwdevice_get_type_name(type));
//		fprintf(stderr, "\n");
//		return -1;
//	}
//
//	/* open the input file */
//	if (avformat_open_input(&input_ctx, argv[2], NULL, NULL) != 0) {
//		fprintf(stderr, "Cannot open input file '%s'\n", argv[2]);
//		return -1;
//	}
//
//	if (avformat_find_stream_info(input_ctx, NULL) < 0) {
//		fprintf(stderr, "Cannot find input stream information.\n");
//		return -1;
//	}
//
//	/* find the video stream information */
//	ret = av_find_best_stream(input_ctx, AVMEDIA_TYPE_VIDEO, -1, -1, &decoder, 0);
//	if (ret < 0) {
//		fprintf(stderr, "Cannot find a video stream in the input file\n");
//		return -1;
//	}
//	video_stream = ret;
//
//	//查找到对应硬件类型解码后的数据格式
//	for (i = 0;; i++) {
//		const AVCodecHWConfig* config = avcodec_get_hw_config(decoder, i);
//		if (!config) {
//			fprintf(stderr, "Decoder %s does not support device type %s.\n",
//				decoder->name, av_hwdevice_get_type_name(type));
//			return -1;
//		}
//		if (config->methods & AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX &&
//			config->device_type == type) {
//			hw_pix_fmt = config->pix_fmt;
//			break;
//		}
//	}
//
//	if (!(decoder_ctx = avcodec_alloc_context3(decoder)))
//		return AVERROR(ENOMEM);
//
//	video = input_ctx->streams[video_stream];
//	if (avcodec_parameters_to_context(decoder_ctx, video->codecpar) < 0)
//		return -1;
//
//	decoder_ctx->get_format = get_hw_format;
//
//	//硬件加速初始化	
//	if (hw_decoder_init(decoder_ctx, type) < 0)
//		return -1;
//
//	if ((ret = avcodec_open2(decoder_ctx, decoder, NULL)) < 0) {
//		fprintf(stderr, "Failed to open codec for stream #%u\n", video_stream);
//		return -1;
//	}
//
//	/* open the file to dump raw data */
//	output_file = fopen(argv[3], "w+b");
//
//	/* actual decoding and dump the raw data */
//	while (ret >= 0) {
//		if ((ret = av_read_frame(input_ctx, &packet)) < 0)
//			break;
//
//		if (video_stream == packet.stream_index)
//			ret = decode_write(decoder_ctx, &packet); //解码并dump文件
//
//		av_packet_unref(&packet);
//	}
//
//	/* flush the decoder */
//	packet.data = NULL;
//	packet.size = 0;
//	ret = decode_write(decoder_ctx, &packet);
//	av_packet_unref(&packet);
//
//	if (output_file)
//		fclose(output_file);
//	avcodec_free_context(&decoder_ctx);
//	avformat_close_input(&input_ctx);
//	av_buffer_unref(&hw_device_ctx);
//
//	return 0;
//}