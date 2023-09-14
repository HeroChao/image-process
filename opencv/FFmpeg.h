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
	return 0;
}

void ffmpeg_transform(const string url) {
	avformat_network_init();

	AVFormatContext* inVFmtCtx = NULL, * outFmtCtx = NULL;
	int frame_index = 0;//统计帧数
	int inVStreamIndex = -1, outVStreamIndex = -1;//输入输出视频流在文件中的索引位置
	const string inVFileName = "rtsp://"+url+":8554/test";
	const char* outFileName = "video.mp4";

	//======================输入部分============================//

	inVFmtCtx = avformat_alloc_context();//初始化内存

	//打开输入文件
	//打开一个文件并解析。可解析的内容包括：视频流、音频流、视频流参数、音频流参数、视频帧索引。
	//参数一：AVFormatContext **ps, 格式化的上下文（由avformat_alloc_context分配）的指针。
	//参数二：要打开的流的url,地址最终会存入到AVFormatContext结构体当中。
	//参数三：指定输入的封装格式。一般传NULL，由FFmpeg自行探测。
	//参数四：包含AVFormatContext和demuxer私有选项的字典。返回时，此参数将被销毁并替换为包含找不到的选项
	if (avformat_open_input(&inVFmtCtx, inVFileName.c_str(), NULL, NULL) < 0) {
		printf("Cannot open input file.\n");
		return;
	}

	//查找输入文件中的流
	/*avformat_find_stream_info函数*/
	//参数一：媒体文件上下文。
	//参数二：字典，一些配置选项。      /*媒体句柄*/
	if (avformat_find_stream_info(inVFmtCtx, NULL) < 0) {
		printf("Cannot find stream info in input file.\n");
		return;
	}

	//查找视频流在文件中的位置
	for (size_t i = 0; i < inVFmtCtx->nb_streams; i++) {//nb_streams 视音频流的个数
		//streams ：输入视频的AVStream []数组  codec：该流对应的AVCodecContext
		if (inVFmtCtx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {//streams 视音频流
			inVStreamIndex = (int)i;
			break;
		}
	}

	AVCodecParameters* codecPara = inVFmtCtx->streams[inVStreamIndex]->codecpar;//输入视频流的编码参数


	printf("===============Input information========>\n");
	av_dump_format(inVFmtCtx, 0, inVFileName.c_str(), 0); //输出视频信息
	printf("===============Input information========<\n");


	//=====================输出部分=========================//
	//打开输出文件并填充格式数据
	//参数一：函数调用成功之后创建的AVFormatContext结构体。
	//参数二：指定AVFormatContext中的AVOutputFormat，确定输出格式。指定为NULL，设定后两个参数（format_name或者filename）由FFmpeg猜测输出格式。。
	//参数三：使用该参数需要自己手动获取AVOutputFormat，相对于使用后两个参数来说要麻烦一些。
	//参数四：指定输出格式的名称。根据格式名称，FFmpeg会推测输出格式。输出格式可以是“flv”，“mkv”等等。
	if (avformat_alloc_output_context2(&outFmtCtx, NULL, NULL, outFileName) < 0) {
		printf("Cannot alloc output file context.\n");
		return;
	}


	//打开输出文件并填充数据
	if (avio_open(&outFmtCtx->pb, outFileName, AVIO_FLAG_READ_WRITE) < 0) {
		printf("output file open failed.\n");
		return;
	}


	//在输出的mp4文件中创建一条视频流
	AVStream* outVStream = avformat_new_stream(outFmtCtx, NULL);//记录视频流通道数目。存储视频流通道。
	if (!outVStream) {
		printf("Failed allocating output stream.\n");
		return;
	}

	outVStream->time_base.den = 25;//AVRational这个结构标识一个分数，num为分数，den为分母(时间的刻度)
	outVStream->time_base.num = 1;
	outVStreamIndex = outVStream->index;



	//查找编码器
	//参数一：id请求的编码器的AVCodecID
	//参数二：如果找到一个编码器，则为NULL。
	//H264/H265码流后，再调用avcodec_find_decoder解码后，再写入到/MP4文件中去
	const AVCodec* outCodec = avcodec_find_decoder(codecPara->codec_id);
	if (outCodec == NULL) {
		printf("Cannot find any encoder.\n");
		return;
	}


	//从输入的h264编码器数据复制一份到输出文件的编码器中
	AVCodecContext* outCodecCtx = avcodec_alloc_context3(outCodec); //申请AVCodecContext空间。需要传递一个编码器，也可以不传，但不会包含编码器。
	//AVCodecParameters与AVCodecContext里的参数有很多相同的
	AVCodecParameters* outCodecPara = outFmtCtx->streams[outVStream->index]->codecpar;

	//avcodec_parameters_copy()来copyAVCodec的上下文。
	if (avcodec_parameters_copy(outCodecPara, codecPara) < 0) {
		printf("Cannot copy codec para.\n");
		return;
	}

	//基于编解码器提供的编解码参数设置编解码器上下文参数
	//参数一：要设置参数的编解码器上下文
	//参数二：媒体流的参数信息 , 包含 码率 , 宽度 , 高度 , 采样率 等参数信息
	if (avcodec_parameters_to_context(outCodecCtx, outCodecPara) < 0) {
		printf("Cannot alloc codec ctx from para.\n");
		return;
	}

	//设置编码器参数(不同参数对视频编质量或大小的影响)
	/*outCodecCtx->time_base.den=25;
	outCodecCtx->time_base.num=1;*/
	outCodecCtx->bit_rate = 0;//目标的码率，即采样的码率；显然，采样码率越大，视频大小越大  比特率
	outCodecCtx->time_base.num = 1;//下面两行：一秒钟25帧
	outCodecCtx->time_base.den = 15;
	outCodecCtx->frame_number = 1;//每包一个视频帧


	//打开输出文件需要的编码器
	if (avcodec_open2(outCodecCtx, outCodec, NULL) < 0) {
		printf("Cannot open output codec.\n");
		return;
	}



	printf("============Output Information=============>\n");
	av_dump_format(outFmtCtx, 0, outFileName, 1);//输出视频信息
	printf("============Output Information=============<\n");


	//写入文件头
	if (avformat_write_header(outFmtCtx, NULL) < 0) {
		printf("Cannot write header to file.\n");
		return;
	}

	//===============编码部分===============//
	//AVPacket 数据结构 显示时间戳（pts）、解码时间戳（dts）、数据时长，所在媒体流的索引等
	AVPacket* pkt = av_packet_alloc();
	//存储每一个视频/音频流信息的结构体
	AVStream* inVStream = inVFmtCtx->streams[inVStreamIndex];
	int i = 0;
	//循环读取每一帧直到读完 从媒体流中读取帧填充到填充到Packet的数据缓存空间
	while (av_read_frame(inVFmtCtx, pkt) >= 0) {//循环读取每一帧直到读完
		i++;
		if (i > 375) {
			break;
		}
		pkt->dts = 0;//不加这个时间戳会出问题，时间戳比之前小的话 FFmpeg会选择丢弃视频包，现在给视频包打时间戳可以重0开始依次递增。
		if (pkt->stream_index == inVStreamIndex) {//确保处理的是视频流 stream_index标识该AVPacket所属的视频/音频流。
			//FIXME：No PTS (Example: Raw H.264)
			//Simple Write PTS
			//如果当前处理帧的显示时间戳为0或者没有等等不是正常值
			if (pkt->pts == AV_NOPTS_VALUE) {
				printf("frame_index:%d\n", frame_index);

				//Write PTS时间 刻度
				AVRational time_base1 = inVStream->time_base;

				//Duration between 2 frames (us) 时长
				//AV_TIME_BASE 时间基
				//av_q2d(AVRational);该函数负责把AVRational结构转换成double，通过这个函数可以计算出某一帧在视频中的时间位置
				//r_frame_rate
				int64_t calc_duration = (double)AV_TIME_BASE / av_q2d(inVStream->r_frame_rate);
				//Parameters参数
				pkt->pts = (double)(frame_index * calc_duration) / (double)(av_q2d(time_base1) * AV_TIME_BASE);
				pkt->dts = pkt->pts;
				pkt->duration = (double)calc_duration / (double)(av_q2d(time_base1) * AV_TIME_BASE);
				frame_index++;
			}
			//Convert PTS/DTS
			//AVPacket
			// pts 显示时间戳
			// dts 解码时间戳
			// duration 数据的时长，以所属媒体流的时间基准为单位
			// pos 数据在媒体流中的位置，未知则值为-1
			// 标识该AVPacket所属的视频/音频流。
			pkt->pts = av_rescale_q_rnd(pkt->pts, inVStream->time_base, outVStream->time_base, (enum AVRounding)(AV_ROUND_NEAR_INF | AV_ROUND_PASS_MINMAX));
			pkt->dts = av_rescale_q_rnd(pkt->dts, inVStream->time_base, outVStream->time_base, (enum AVRounding)(AV_ROUND_NEAR_INF | AV_ROUND_PASS_MINMAX));
			pkt->duration = av_rescale_q(pkt->duration, inVStream->time_base, outVStream->time_base);
			pkt->pos = -1;
			pkt->stream_index = outVStreamIndex;
			//Write
			if (av_interleaved_write_frame(outFmtCtx, pkt) < 0) {
				printf("Error muxing packet\n");
				break;
			}
			//处理完压缩数据之后，并且在进入下一次循环之前，
			//记得使用 av_packet_unref 来释放已经分配的AVPacket->data缓冲区。
			av_packet_unref(pkt);
		}
	}

	av_write_trailer(outFmtCtx);

	//=================释放所有指针=======================
	av_packet_free(&pkt);//堆栈上数据缓存空间
	av_free(inVStream);//存储每一个视频/音频流信息的结构体
	av_free(outVStream);//在输出的mp4文件中创建一条视频流
	avformat_close_input(&outFmtCtx);//关闭一个AVFormatContext
	avcodec_close(outCodecCtx);
	avcodec_free_context(&outCodecCtx);
	avcodec_parameters_free(&outCodecPara);
	avcodec_parameters_free(&codecPara);
	avformat_close_input(&inVFmtCtx);//关闭一个AVFormatContext
	avformat_free_context(inVFmtCtx);//销毁函数
	avio_close(outFmtCtx->pb);
}