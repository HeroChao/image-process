#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
// 引用 FFmpeg C 头文件
extern "C"
{
#include<libavutil/opt.h>
#include<libavutil/channel_layout.h>
#include<libavutil/common.h>
#include<libavutil/imgutils.h>
#include<libavutil/mathematics.h>
#include<libavutil/samplefmt.h>
#include<libavutil/time.h>
#include<libavutil/fifo.h>
#include<libavcodec/avcodec.h>
#include<libavformat/avformat.h>
#include<libavformat/avio.h>
#include<libavfilter/avfilter.h>
#include<libavfilter/buffersink.h>
#include<libavfilter/buffersrc.h>
#include<libswscale/swscale.h>
#include<libswresample/swresample.h>
}

static AVBufferRef* hw_device_ctx = nullptr;
static enum AVPixelFormat hw_pix_fmt;
static FILE* output_file = NULL;

//硬件加速初始化	
static int hw_decoder_init(AVCodecContext* ctx, const enum AVHWDeviceType type)
{
	int err = 0;
	//创建一个硬件设备上下文
	if ((err = av_hwdevice_ctx_create(&hw_device_ctx, type,
		NULL, NULL, 0)) < 0) {
		fprintf(stderr, "Failed to create specified HW device.\n");
		return err;
	}
	ctx->hw_device_ctx = av_buffer_ref(hw_device_ctx);

	return err;
}

//获取GPU硬件解码帧的格式
static enum AVPixelFormat get_hw_format(AVCodecContext* ctx,
	const enum AVPixelFormat* pix_fmts)
{
	const enum AVPixelFormat* p;

	for (p = pix_fmts; *p != -1; p++) {
		if (*p == hw_pix_fmt)
			return *p;
	}

	fprintf(stderr, "Failed to get HW surface format.\n");
	return AV_PIX_FMT_NONE;
}

static cv::Mat AVFrameToCvMat(AVFrame* input_avframe)
{
	//AVframe转为Mat类型；
	int image_width = input_avframe->width;
	int image_height = input_avframe->height;

	cv::Mat resMat(image_height, image_width, CV_8UC3);
	int cvLinesizes[1];
	cvLinesizes[0] = resMat.step1();

	SwsContext* avFrameToOpenCVBGRSwsContext = sws_getContext(
		image_width,
		image_height,
		AVPixelFormat::AV_PIX_FMT_YUV420P,
		image_width,
		image_height,
		AVPixelFormat::AV_PIX_FMT_BGR24,
		SWS_FAST_BILINEAR,
		nullptr, nullptr, nullptr
	);

	sws_scale(avFrameToOpenCVBGRSwsContext,
		input_avframe->data,
		input_avframe->linesize,
		0,
		image_height,
		&resMat.data,
		cvLinesizes);

	if (avFrameToOpenCVBGRSwsContext != nullptr)
	{
		sws_freeContext(avFrameToOpenCVBGRSwsContext);
		avFrameToOpenCVBGRSwsContext = nullptr;
	}

	return resMat;
}

//解码后数据格式转换，GPU到CPU拷贝，YUV数据dump到文件
static int decode_write(AVCodecContext* avctx, AVPacket* packet)
{
	AVFrame* frame = NULL, * sw_frame = NULL;
	AVFrame* tmp_frame = NULL;
	uint8_t* buffer = NULL;
	int size;
	int ret = 0;

	ret = avcodec_send_packet(avctx, packet);
	if (ret < 0) {
		fprintf(stderr, "Error during decoding\n");
		return ret;
	}

	while (1) {
		if (!(frame = av_frame_alloc()) || !(sw_frame = av_frame_alloc())) {
			fprintf(stderr, "Can not alloc frame\n");
			ret = AVERROR(ENOMEM);
			goto fail;
		}

		ret = avcodec_receive_frame(avctx, frame);
		if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
			av_frame_free(&frame);
			av_frame_free(&sw_frame);
			return 0;
		}
		else if (ret < 0) {
			fprintf(stderr, "Error while decoding\n");
			goto fail;
		}

		if (frame->format == hw_pix_fmt) {
			/* 将解码后的数据从GPU内存存格式转为CPU内存格式，并完成GPU到CPU内存的拷贝*/
			if ((ret = av_hwframe_transfer_data(sw_frame, frame, 0)) < 0) {
				fprintf(stderr, "Error transferring the data to system memory\n");
				goto fail;
			}
			tmp_frame = sw_frame;
		}
		else
			tmp_frame = frame;
		//计算一张YUV图需要的内存 大小
		size = av_image_get_buffer_size((AVPixelFormat)tmp_frame->format, tmp_frame->width,
			tmp_frame->height, 1);
		//分配内存
		buffer = (uint8_t*)av_malloc(size);
		if (!buffer) {
			fprintf(stderr, "Can not alloc buffer\n");
			ret = AVERROR(ENOMEM);
			goto fail;
		}
		//将图片数据拷贝的buffer中(按行拷贝)
		ret = av_image_copy_to_buffer(buffer, size,
			(const uint8_t* const*)tmp_frame->data,
			(const int*)tmp_frame->linesize, (AVPixelFormat)tmp_frame->format,
			tmp_frame->width, tmp_frame->height, 1);
		if (ret < 0) {
			fprintf(stderr, "Can not copy image to buffer\n");
			goto fail;
		}
		//buffer数据dump到文件
		if ((ret = fwrite(buffer, 1, size, output_file)) < 0) {
			fprintf(stderr, "Failed to dump raw data.\n");
			goto fail;
		}

	fail:
		av_frame_free(&frame);
		av_frame_free(&sw_frame);
		av_freep(&buffer);
		if (ret < 0)
			return ret;
	}
}

int ffmpeg_decodec(const char* url) {
	avformat_network_init();
	AVFormatContext* formatContext = avformat_alloc_context();
	if (avformat_open_input(&formatContext, url, NULL, NULL) != 0) {
		// 打开失败，进行错误处理
		return -1;
	}
	if (avformat_find_stream_info(formatContext, NULL) < 0) {
		// 未找到流信息，进行错误处理
		return -1;
	}

	int videoStreamIndex = -1;
	AVCodecParameters* codecParameters = nullptr;
	for (int i = 0; i < formatContext->nb_streams; i++) {
		if (formatContext->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
			videoStreamIndex = i;
			codecParameters = formatContext->streams[i]->codecpar;
			break;
		}
	}

	if (videoStreamIndex == -1 || codecParameters == nullptr) {
		// 未找到视频流或编解码参数错误，进行错误处理
		cout<<"NO streams" << endl;
		return -1;
	}

	const AVCodec* codec = avcodec_find_decoder(codecParameters->codec_id);
	AVCodecContext* codecContext = avcodec_alloc_context3(codec);
	avcodec_parameters_to_context(codecContext, codecParameters);
	avcodec_open2(codecContext, codec, NULL);

	AVFrame* frame = av_frame_alloc();
	AVPacket packet;
	while (av_read_frame(formatContext, &packet) >= 0) {
		if (packet.stream_index == videoStreamIndex) {
			avcodec_send_packet(codecContext, &packet);
			int ret = avcodec_receive_frame(codecContext, frame);
			if (ret == 0) {
				cv::Mat image(frame->height, frame->width, CV_8UC3);
				SwsContext* swsContext = sws_getContext(frame->width, frame->height, codecContext->pix_fmt,
					frame->width, frame->height, AV_PIX_FMT_BGR24,
					0, NULL, NULL, NULL);
				uint8_t* destData[1] = { image.data };
				int destLinesize[1] = { image.step };
				sws_scale(swsContext, frame->data, frame->linesize, 0, frame->height, destData, destLinesize);
				cv::imshow("Frame", image);
				cv::waitKey(1);
				sws_freeContext(swsContext);
			}
		}
		av_packet_unref(&packet);
	}
	av_frame_free(&frame);
	avcodec_close(codecContext);
	avformat_close_input(&formatContext);
	return 0;
}

int ffmpeg_encodec(const char* url) {
	avformat_network_init();
	cv::VideoCapture capture(0);
	cv::Mat frame;
	const AVOutputFormat* outFormat = av_guess_format("rtsp", NULL, NULL);
	AVFormatContext* outFormatContext;
	if (avformat_alloc_output_context2(&outFormatContext, outFormat, "rtsp", url) < 0) {
		std::cout << "无法创建输出上下文" << std::endl;
		return -1;
	}
	AVStream* videoSt = avformat_new_stream(outFormatContext, NULL);
	if (!videoSt) {
		std::cout << "无法创建视频流" << std::endl;
		return -1;
	}
	const AVCodec* codec = avcodec_find_encoder(outFormat->video_codec);
	if (!codec) {
		std::cout << "找不到视频编码器" << std::endl;
		return -1;
	}
	AVCodecContext* codecContext = avcodec_alloc_context3(codec);
	if (!codecContext) {
		std::cout << "无法分配编码器上下文" << std::endl;
		return -1;
	}
	codecContext->codec_id = outFormat->video_codec;
	codecContext->codec_type = AVMEDIA_TYPE_VIDEO;
	codecContext->pix_fmt = AV_PIX_FMT_YUV420P;
	codecContext->width = frame.cols;
	codecContext->height = frame.rows;
	codecContext->time_base = AVRational{ 1, 30 };

	if (outFormatContext->oformat->flags & AVFMT_GLOBALHEADER)
		codecContext->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
	if (avcodec_parameters_from_context(videoSt->codecpar, codecContext) < 0) {
		std::cout << "无法设置编码器上下文参数" << std::endl;
		return -1;
	}
	if (avcodec_open2(codecContext, codec, nullptr) < 0) {
		std::cout << "无法打开编码器" << std::endl;
		return -1;
	}
	if (avio_open(&outFormatContext->pb, url, AVIO_FLAG_WRITE) < 0)
	{
		std::cout << "无法打开RTSP输出流" << std::endl;
		return -1;
	}

	if (avformat_write_header(outFormatContext, NULL) < 0)
	{
		std::cout << "无法写入RTSP头部信息" << std::endl;
		return -1;
	}
	while (capture.read(frame))
	{
		cv::Mat rgbFrame;
		cv::cvtColor(frame, rgbFrame, cv::COLOR_BGR2RGB);

		AVPacket pkt;
		av_init_packet(&pkt);
		pkt.data = NULL;
		pkt.size = 0;

		cv::Mat yuvFrame;
		cv::cvtColor(rgbFrame, yuvFrame, cv::COLOR_RGB2YUV_I420);
		pkt.data = yuvFrame.data;
		pkt.size = yuvFrame.total() * yuvFrame.elemSize();

		if (av_write_frame(outFormatContext, &pkt) < 0)
		{
			std::cout << "无法写入RTSP帧数据" << std::endl;
			break;
		}

		av_packet_unref(&pkt);
	}
	av_write_trailer(outFormatContext);
	avio_close(outFormatContext->pb);
	avformat_free_context(outFormatContext);
	capture.release();

}