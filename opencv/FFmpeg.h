#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
// ���� FFmpeg C ͷ�ļ�
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

//Ӳ�����ٳ�ʼ��	
static int hw_decoder_init(AVCodecContext* ctx, const enum AVHWDeviceType type)
{
	int err = 0;
	//����һ��Ӳ���豸������
	if ((err = av_hwdevice_ctx_create(&hw_device_ctx, type,
		NULL, NULL, 0)) < 0) {
		fprintf(stderr, "Failed to create specified HW device.\n");
		return err;
	}
	ctx->hw_device_ctx = av_buffer_ref(hw_device_ctx);

	return err;
}

//��ȡGPUӲ������֡�ĸ�ʽ
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
	//AVframeתΪMat���ͣ�
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

//��������ݸ�ʽת����GPU��CPU������YUV����dump���ļ�
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
			/* �����������ݴ�GPU�ڴ���ʽתΪCPU�ڴ��ʽ�������GPU��CPU�ڴ�Ŀ���*/
			if ((ret = av_hwframe_transfer_data(sw_frame, frame, 0)) < 0) {
				fprintf(stderr, "Error transferring the data to system memory\n");
				goto fail;
			}
			tmp_frame = sw_frame;
		}
		else
			tmp_frame = frame;
		//����һ��YUVͼ��Ҫ���ڴ� ��С
		size = av_image_get_buffer_size((AVPixelFormat)tmp_frame->format, tmp_frame->width,
			tmp_frame->height, 1);
		//�����ڴ�
		buffer = (uint8_t*)av_malloc(size);
		if (!buffer) {
			fprintf(stderr, "Can not alloc buffer\n");
			ret = AVERROR(ENOMEM);
			goto fail;
		}
		//��ͼƬ���ݿ�����buffer��(���п���)
		ret = av_image_copy_to_buffer(buffer, size,
			(const uint8_t* const*)tmp_frame->data,
			(const int*)tmp_frame->linesize, (AVPixelFormat)tmp_frame->format,
			tmp_frame->width, tmp_frame->height, 1);
		if (ret < 0) {
			fprintf(stderr, "Can not copy image to buffer\n");
			goto fail;
		}
		//buffer����dump���ļ�
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
		// ��ʧ�ܣ����д�����
		return -1;
	}
	if (avformat_find_stream_info(formatContext, NULL) < 0) {
		// δ�ҵ�����Ϣ�����д�����
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
		// δ�ҵ���Ƶ��������������󣬽��д�����
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
		std::cout << "�޷��������������" << std::endl;
		return -1;
	}
	AVStream* videoSt = avformat_new_stream(outFormatContext, NULL);
	if (!videoSt) {
		std::cout << "�޷�������Ƶ��" << std::endl;
		return -1;
	}
	const AVCodec* codec = avcodec_find_encoder(outFormat->video_codec);
	if (!codec) {
		std::cout << "�Ҳ�����Ƶ������" << std::endl;
		return -1;
	}
	AVCodecContext* codecContext = avcodec_alloc_context3(codec);
	if (!codecContext) {
		std::cout << "�޷����������������" << std::endl;
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
		std::cout << "�޷����ñ����������Ĳ���" << std::endl;
		return -1;
	}
	if (avcodec_open2(codecContext, codec, nullptr) < 0) {
		std::cout << "�޷��򿪱�����" << std::endl;
		return -1;
	}
	if (avio_open(&outFormatContext->pb, url, AVIO_FLAG_WRITE) < 0)
	{
		std::cout << "�޷���RTSP�����" << std::endl;
		return -1;
	}

	if (avformat_write_header(outFormatContext, NULL) < 0)
	{
		std::cout << "�޷�д��RTSPͷ����Ϣ" << std::endl;
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
			std::cout << "�޷�д��RTSP֡����" << std::endl;
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
	int frame_index = 0;//ͳ��֡��
	int inVStreamIndex = -1, outVStreamIndex = -1;//���������Ƶ�����ļ��е�����λ��
	const string inVFileName = "rtsp://"+url+":8554/test";
	const char* outFileName = "video.mp4";

	//======================���벿��============================//

	inVFmtCtx = avformat_alloc_context();//��ʼ���ڴ�

	//�������ļ�
	//��һ���ļ����������ɽ��������ݰ�������Ƶ������Ƶ������Ƶ����������Ƶ����������Ƶ֡������
	//����һ��AVFormatContext **ps, ��ʽ���������ģ���avformat_alloc_context���䣩��ָ�롣
	//��������Ҫ�򿪵�����url,��ַ���ջ���뵽AVFormatContext�ṹ�嵱�С�
	//��������ָ������ķ�װ��ʽ��һ�㴫NULL����FFmpeg����̽�⡣
	//�����ģ�����AVFormatContext��demuxer˽��ѡ����ֵ䡣����ʱ���˲����������ٲ��滻Ϊ�����Ҳ�����ѡ��
	if (avformat_open_input(&inVFmtCtx, inVFileName.c_str(), NULL, NULL) < 0) {
		printf("Cannot open input file.\n");
		return;
	}

	//���������ļ��е���
	/*avformat_find_stream_info����*/
	//����һ��ý���ļ������ġ�
	//���������ֵ䣬һЩ����ѡ�      /*ý����*/
	if (avformat_find_stream_info(inVFmtCtx, NULL) < 0) {
		printf("Cannot find stream info in input file.\n");
		return;
	}

	//������Ƶ�����ļ��е�λ��
	for (size_t i = 0; i < inVFmtCtx->nb_streams; i++) {//nb_streams ����Ƶ���ĸ���
		//streams ��������Ƶ��AVStream []����  codec��������Ӧ��AVCodecContext
		if (inVFmtCtx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {//streams ����Ƶ��
			inVStreamIndex = (int)i;
			break;
		}
	}

	AVCodecParameters* codecPara = inVFmtCtx->streams[inVStreamIndex]->codecpar;//������Ƶ���ı������


	printf("===============Input information========>\n");
	av_dump_format(inVFmtCtx, 0, inVFileName.c_str(), 0); //�����Ƶ��Ϣ
	printf("===============Input information========<\n");


	//=====================�������=========================//
	//������ļ�������ʽ����
	//����һ���������óɹ�֮�󴴽���AVFormatContext�ṹ�塣
	//��������ָ��AVFormatContext�е�AVOutputFormat��ȷ�������ʽ��ָ��ΪNULL���趨������������format_name����filename����FFmpeg�²������ʽ����
	//��������ʹ�øò�����Ҫ�Լ��ֶ���ȡAVOutputFormat�������ʹ�ú�����������˵Ҫ�鷳һЩ��
	//�����ģ�ָ�������ʽ�����ơ����ݸ�ʽ���ƣ�FFmpeg���Ʋ������ʽ�������ʽ�����ǡ�flv������mkv���ȵȡ�
	if (avformat_alloc_output_context2(&outFmtCtx, NULL, NULL, outFileName) < 0) {
		printf("Cannot alloc output file context.\n");
		return;
	}


	//������ļ����������
	if (avio_open(&outFmtCtx->pb, outFileName, AVIO_FLAG_READ_WRITE) < 0) {
		printf("output file open failed.\n");
		return;
	}


	//�������mp4�ļ��д���һ����Ƶ��
	AVStream* outVStream = avformat_new_stream(outFmtCtx, NULL);//��¼��Ƶ��ͨ����Ŀ���洢��Ƶ��ͨ����
	if (!outVStream) {
		printf("Failed allocating output stream.\n");
		return;
	}

	outVStream->time_base.den = 25;//AVRational����ṹ��ʶһ��������numΪ������denΪ��ĸ(ʱ��Ŀ̶�)
	outVStream->time_base.num = 1;
	outVStreamIndex = outVStream->index;



	//���ұ�����
	//����һ��id����ı�������AVCodecID
	//������������ҵ�һ������������ΪNULL��
	//H264/H265�������ٵ���avcodec_find_decoder�������д�뵽/MP4�ļ���ȥ
	const AVCodec* outCodec = avcodec_find_decoder(codecPara->codec_id);
	if (outCodec == NULL) {
		printf("Cannot find any encoder.\n");
		return;
	}


	//�������h264���������ݸ���һ�ݵ�����ļ��ı�������
	AVCodecContext* outCodecCtx = avcodec_alloc_context3(outCodec); //����AVCodecContext�ռ䡣��Ҫ����һ����������Ҳ���Բ����������������������
	//AVCodecParameters��AVCodecContext��Ĳ����кܶ���ͬ��
	AVCodecParameters* outCodecPara = outFmtCtx->streams[outVStream->index]->codecpar;

	//avcodec_parameters_copy()��copyAVCodec�������ġ�
	if (avcodec_parameters_copy(outCodecPara, codecPara) < 0) {
		printf("Cannot copy codec para.\n");
		return;
	}

	//���ڱ�������ṩ�ı����������ñ�����������Ĳ���
	//����һ��Ҫ���ò����ı������������
	//��������ý�����Ĳ�����Ϣ , ���� ���� , ��� , �߶� , ������ �Ȳ�����Ϣ
	if (avcodec_parameters_to_context(outCodecCtx, outCodecPara) < 0) {
		printf("Cannot alloc codec ctx from para.\n");
		return;
	}

	//���ñ���������(��ͬ��������Ƶ���������С��Ӱ��)
	/*outCodecCtx->time_base.den=25;
	outCodecCtx->time_base.num=1;*/
	outCodecCtx->bit_rate = 0;//Ŀ������ʣ������������ʣ���Ȼ����������Խ����Ƶ��СԽ��  ������
	outCodecCtx->time_base.num = 1;//�������У�һ����25֡
	outCodecCtx->time_base.den = 15;
	outCodecCtx->frame_number = 1;//ÿ��һ����Ƶ֡


	//������ļ���Ҫ�ı�����
	if (avcodec_open2(outCodecCtx, outCodec, NULL) < 0) {
		printf("Cannot open output codec.\n");
		return;
	}



	printf("============Output Information=============>\n");
	av_dump_format(outFmtCtx, 0, outFileName, 1);//�����Ƶ��Ϣ
	printf("============Output Information=============<\n");


	//д���ļ�ͷ
	if (avformat_write_header(outFmtCtx, NULL) < 0) {
		printf("Cannot write header to file.\n");
		return;
	}

	//===============���벿��===============//
	//AVPacket ���ݽṹ ��ʾʱ�����pts��������ʱ�����dts��������ʱ��������ý������������
	AVPacket* pkt = av_packet_alloc();
	//�洢ÿһ����Ƶ/��Ƶ����Ϣ�Ľṹ��
	AVStream* inVStream = inVFmtCtx->streams[inVStreamIndex];
	int i = 0;
	//ѭ����ȡÿһֱ֡������ ��ý�����ж�ȡ֡��䵽��䵽Packet�����ݻ���ռ�
	while (av_read_frame(inVFmtCtx, pkt) >= 0) {//ѭ����ȡÿһֱ֡������
		i++;
		if (i > 375) {
			break;
		}
		pkt->dts = 0;//�������ʱ���������⣬ʱ�����֮ǰС�Ļ� FFmpeg��ѡ������Ƶ�������ڸ���Ƶ����ʱ���������0��ʼ���ε�����
		if (pkt->stream_index == inVStreamIndex) {//ȷ�����������Ƶ�� stream_index��ʶ��AVPacket��������Ƶ/��Ƶ����
			//FIXME��No PTS (Example: Raw H.264)
			//Simple Write PTS
			//�����ǰ����֡����ʾʱ���Ϊ0����û�еȵȲ�������ֵ
			if (pkt->pts == AV_NOPTS_VALUE) {
				printf("frame_index:%d\n", frame_index);

				//Write PTSʱ�� �̶�
				AVRational time_base1 = inVStream->time_base;

				//Duration between 2 frames (us) ʱ��
				//AV_TIME_BASE ʱ���
				//av_q2d(AVRational);�ú��������AVRational�ṹת����double��ͨ������������Լ����ĳһ֡����Ƶ�е�ʱ��λ��
				//r_frame_rate
				int64_t calc_duration = (double)AV_TIME_BASE / av_q2d(inVStream->r_frame_rate);
				//Parameters����
				pkt->pts = (double)(frame_index * calc_duration) / (double)(av_q2d(time_base1) * AV_TIME_BASE);
				pkt->dts = pkt->pts;
				pkt->duration = (double)calc_duration / (double)(av_q2d(time_base1) * AV_TIME_BASE);
				frame_index++;
			}
			//Convert PTS/DTS
			//AVPacket
			// pts ��ʾʱ���
			// dts ����ʱ���
			// duration ���ݵ�ʱ����������ý������ʱ���׼Ϊ��λ
			// pos ������ý�����е�λ�ã�δ֪��ֵΪ-1
			// ��ʶ��AVPacket��������Ƶ/��Ƶ����
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
			//������ѹ������֮�󣬲����ڽ�����һ��ѭ��֮ǰ��
			//�ǵ�ʹ�� av_packet_unref ���ͷ��Ѿ������AVPacket->data��������
			av_packet_unref(pkt);
		}
	}

	av_write_trailer(outFmtCtx);

	//=================�ͷ�����ָ��=======================
	av_packet_free(&pkt);//��ջ�����ݻ���ռ�
	av_free(inVStream);//�洢ÿһ����Ƶ/��Ƶ����Ϣ�Ľṹ��
	av_free(outVStream);//�������mp4�ļ��д���һ����Ƶ��
	avformat_close_input(&outFmtCtx);//�ر�һ��AVFormatContext
	avcodec_close(outCodecCtx);
	avcodec_free_context(&outCodecCtx);
	avcodec_parameters_free(&outCodecPara);
	avcodec_parameters_free(&codecPara);
	avformat_close_input(&inVFmtCtx);//�ر�һ��AVFormatContext
	avformat_free_context(inVFmtCtx);//���ٺ���
	avio_close(outFmtCtx->pb);
}