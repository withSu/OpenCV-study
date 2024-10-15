#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <malloc.h>


#include<stdio.h>
#include <opencv2/opencv.hpp>

using namespace cv;

typedef struct {
	int r, g, b;
}int_rgb;


int** IntAlloc2(int height, int width)
{
	int** tmp;
	tmp = (int**)calloc(height, sizeof(int*));
	for (int i = 0; i < height; i++)
		tmp[i] = (int*)calloc(width, sizeof(int));
	return(tmp);
}

void IntFree2(int** image, int height, int width)
{
	for (int i = 0; i < height; i++)
		free(image[i]);

	free(image);
}


float** FloatAlloc2(int height, int width)
{
	float** tmp;
	tmp = (float**)calloc(height, sizeof(float*));
	for (int i = 0; i < height; i++)
		tmp[i] = (float*)calloc(width, sizeof(float));
	return(tmp);
}

void FloatFree2(float** image, int height, int width)
{
	for (int i = 0; i < height; i++)
		free(image[i]);

	free(image);
}

int_rgb** IntColorAlloc2(int height, int width)
{
	int_rgb** tmp;
	tmp = (int_rgb**)calloc(height, sizeof(int_rgb*));
	for (int i = 0; i < height; i++)
		tmp[i] = (int_rgb*)calloc(width, sizeof(int_rgb));
	return(tmp);
}

void IntColorFree2(int_rgb** image, int height, int width)
{
	for (int i = 0; i < height; i++)
		free(image[i]);

	free(image);
}

int** ReadImage(char* name, int* height, int* width)
{
	Mat img = imread(name, IMREAD_GRAYSCALE);
	int** image = (int**)IntAlloc2(img.rows, img.cols);

	*width = img.cols;
	*height = img.rows;

	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++)
			image[i][j] = img.at<unsigned char>(i, j);

	return(image);
}

void WriteImage(char* name, int** image, int height, int width)
{
	Mat img(height, width, CV_8UC1);
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
			img.at<unsigned char>(i, j) = (unsigned char)image[i][j];

	imwrite(name, img);
}


void ImageShow(char* winname, int** image, int height, int width)
{
	Mat img(height, width, CV_8UC1);
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
			img.at<unsigned char>(i, j) = (unsigned char)image[i][j];
	imshow(winname, img);
	waitKey(0);
}



int_rgb** ReadColorImage(char* name, int* height, int* width)
{
	Mat img = imread(name, IMREAD_COLOR);
	int_rgb** image = (int_rgb**)IntColorAlloc2(img.rows, img.cols);

	*width = img.cols;
	*height = img.rows;

	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++) {
			image[i][j].b = img.at<Vec3b>(i, j)[0];
			image[i][j].g = img.at<Vec3b>(i, j)[1];
			image[i][j].r = img.at<Vec3b>(i, j)[2];
		}

	return(image);
}

void WriteColorImage(char* name, int_rgb** image, int height, int width)
{
	Mat img(height, width, CV_8UC3);
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++) {
			img.at<Vec3b>(i, j)[0] = (unsigned char)image[i][j].b;
			img.at<Vec3b>(i, j)[1] = (unsigned char)image[i][j].g;
			img.at<Vec3b>(i, j)[2] = (unsigned char)image[i][j].r;
		}

	imwrite(name, img);
}

void ColorImageShow(char* winname, int_rgb** image, int height, int width)
{
	Mat img(height, width, CV_8UC3);
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++) {
			img.at<Vec3b>(i, j)[0] = (unsigned char)image[i][j].b;
			img.at<Vec3b>(i, j)[1] = (unsigned char)image[i][j].g;
			img.at<Vec3b>(i, j)[2] = (unsigned char)image[i][j].r;
		}
	imshow(winname, img);

}

template <typename _TP>
void ConnectedComponentLabeling(_TP** seg, int height, int width, int** label, int* no_label)
{

	//Mat bw = threshval < 128 ? (img < threshval) : (img > threshval);
	Mat bw(height, width, CV_8U);

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++)
			bw.at<unsigned char>(i, j) = (unsigned char)seg[i][j];
	}
	Mat labelImage(bw.size(), CV_32S);
	*no_label = connectedComponents(bw, labelImage, 8); // 0        Ե        

	(*no_label)--;

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++)
			label[i][j] = labelImage.at<int>(i, j);
	}
}

#define imax(x, y) ((x)>(y) ? x : y)
#define imin(x, y) ((x)<(y) ? x : y)

int BilinearInterpolation(int** image, int width, int height, double x, double y)
{
	int x_int = (int)x;
	int y_int = (int)y;

	int A = image[imin(imax(y_int, 0), height - 1)][imin(imax(x_int, 0), width - 1)];
	int B = image[imin(imax(y_int, 0), height - 1)][imin(imax(x_int + 1, 0), width - 1)];
	int C = image[imin(imax(y_int + 1, 0), height - 1)][imin(imax(x_int, 0), width - 1)];
	int D = image[imin(imax(y_int + 1, 0), height - 1)][imin(imax(x_int + 1, 0), width - 1)];

	double dx = x - x_int;
	double dy = y - y_int;

	double value
		= (1.0 - dx) * (1.0 - dy) * A + dx * (1.0 - dy) * B
		+ (1.0 - dx) * dy * C + dx * dy * D;

	return((int)(value + 0.5));
}


void DrawHistogram(char* comments, int* Hist)
{
	int histSize = 256; /// Establish the number of bins
	// Draw the histograms for B, G and R
	int hist_w = 512; int hist_h = 512;
	int bin_w = cvRound((double)hist_w / histSize);

	Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(255, 255, 255));
	Mat r_hist(histSize, 1, CV_32FC1);
	for (int i = 0; i < histSize; i++)
		r_hist.at<float>(i, 0) = Hist[i];
	/// Normalize the result to [ 0, histImage.rows ]
	normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());

	/// Draw for each channel
	for (int i = 1; i < histSize; i++)
	{
		line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(r_hist.at<float>(i - 1))),
			Point(bin_w * (i), hist_h - cvRound(r_hist.at<float>(i))),
			Scalar(255, 0, 0), 2, 8, 0);
	}

	/// Display
	namedWindow(comments, WINDOW_AUTOSIZE);
	imshow(comments, histImage);

	waitKey(0);

}

/*십자가 만들기*/
void ex0911_1() 
{
	int height = 512, width = 1024;
	int** img = (int**)IntAlloc2(height, width); //이미지 배열을 만든다.

	int y = 256;
	int x = 512;

	//가로줄 추가
	for (int x = 0;x<width; x++)
		img[y][x] = 255;


	//세로줄 추가
	for (int y = 0; y < height; y++)
		img[y][x] = 255;
	


	printf("%d \n", img[100][200]);
	ImageShow((char*)"output", img, height, width);

	IntFree2(img, height, width); //메모리 free를 해준다.
}


/*사각형 만들기*/
void ex0911_2()
{
	int height = 512, width = 1024;
	int** img = (int**)IntAlloc2(height, width); //이미지 배열을 만든다.

	//y= [150,512-400)
	//x= [300, 1024-300)

	

	for (int y = 150; y < height-150; y++)
	{
		for (int x=300; x<width-300;x++)
			img[y][x] = 255;

	}

	printf("%d \n", img[100][200]);
	ImageShow((char*)"output", img, height, width);

	IntFree2(img, height, width); //메모리 free를 해준다.

}


/*호출되는 함수*/
void drawLine(int** imgxx,int height, int width, int y, int x )
{
	for (int y = 150; y < height - 150; y++)
	{
		for (int x = 300; x < width - 300; x++)
			imgxx[y][x] = 255;

	}
}

/*호출하는 함수*/
void ex0911_3()
{
	int height = 512, width = 1024;
	int** img = (int**)IntAlloc2(height, width); //이미지 배열을 만든다.
	int x0=0, y0=0;

	drawLine(img,height, width,  y0,  x0); //호출할떄는 타입을 사용하지 않는다.



	printf("%d \n", img[100][200]);
	ImageShow((char*)"output", img, height, width);

	IntFree2(img, height, width); //메모리 free를 해준다.
	
}


int ex0924_1(){
	int height;
	int width;

	int** img = ReadImage((char*)"./images/barbara.png", &height, &width);
	int** img_out = (int**)IntAlloc2(height, width);
	int threshold = 128;
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			if (img[y][x] > threshold) img_out[y][x] = 255;
			else img_out[y][x] = 0;
		}
	}
	ImageShow((char*)"input", img, height, width);
	ImageShow((char*)"output", img_out, height, width);
	return 0;
}






void Thresholdings(int threshold, int** img, int height, int width, int** img_out)
{
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			if (img[y][x] > threshold) img_out[y][x] = 255;
			else img_out[y][x] = 0;
		}
	}
}


int ex0924_2() {

	int height, width;
	int** img = ReadImage((char*)"./images/lena.png", &height, &width);
	int** img_out = (int**)IntAlloc2(height, width);
	int threshold = 200;

	for (threshold = 50; threshold < 250; threshold += 50) {
		Thresholdings(threshold, img, height, width, img_out);
		ImageShow((char*)"output", img_out, height, width);
		
	}
	ImageShow((char*)"input", img, height, width);

	return 0;

}

void ShiftImage(int value, int** img, int height, int width)
{
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			img[y][x] = img[y][x] + value;
		}
	}
}


#define GetMax(a,b) ((a>b)?a:b)
#define GetMin(a,b) ((a<b)?a:b)

void ClippingImage(int value, int** img, int height, int width, int** img_out)
{
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			/*if (img[y][x] > 255) {
				img_out[y][x] = 255;
			}
			else if (img[y][x] < 0) {
				img_out[y][x] = 0;
			}
			else {
				img_out[y][x] = img[y][x];
			}*/

			/*int A = GetMax(img[y][x], 0);
			int B = GetMin(A, 255);
			img_out[y][x] = B;*/

			img_out[y][x] = GetMin(GetMax(img[y][x], 0), 255);
		}
	}


}

int ex0924_3()
{
	int height, width;
	int** img = ReadImage((char*)"./images/lena.png", &height, &width);
	int** img_out = (int**)IntAlloc2(height, width);
	
	ClippingImage(50, img, height, width, img_out);




	ImageShow((char*)"input", img, height, width);
	ImageShow((char*)"output", img_out, height, width);

	return 0;
}




#define NUM 100

int ex0925_1(void)
{
	int x = 100, y = 200;
	int A, B;

	//A = (x > y) ? x : y;
	//B = (x < y) ? x : y;
	
	A = GetMax(x, 0); //큰값
	B = GetMin(y, 255); //작은값

	return 0;

}



int ex0925_2(void)
{
	int A = 100, B = 200, C = 300;
	//int D = GetMax(A,B);
	int E = GetMax(GetMax(A, B), C);
	return 0;
}

int findMaxvalue(int** img, int height, int width)
{
	int max_value = img [0][0]; 
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			max_value = GetMax(max_value, img[y][x]);
		}
	}

	return max_value;
}
int findMinvalue(int** img, int height, int width)
{
	int min_value = img[0][0];
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			min_value = GetMin(min_value, img[y][x]);
		}
	}

	return min_value;
}



int ex0925_3(void)
{
	int A[7] = { 1,-1,3,8,2,9,10 };

	int max_value = A[0];
	int min_value = A[0];

	for (int n = 1; n < 7; n++)
	{
		max_value = GetMax(max_value, A[n]);
		min_value = GetMin(min_value, A[n]);
	}

	int height, width;
	int** img = ReadImage((char*)"./images/lena.png", &height, &width);
	max_value=findMaxvalue(img, height, width);
	min_value = findMinvalue(img, height, width);


	return 0;
}


void MixingImages(int** img1, int** img2, float alpha, int height, int width, int** img_out)
{


	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			img_out[y][x] = alpha * img1[y][x] + (1.0 - alpha) * img2[y][x];
		}
	}


}

int ex0925_4()
{
	int height, width;
	int** img1 = ReadImage((char*)"./images/lena.png", &height, &width);
	int** img2 = ReadImage((char*)"./images/barbara.png", &height, &width);
	int** img_out = (int**)IntAlloc2(height, width);

	

	for (float alpha = 0.1; alpha < 1.0; alpha += 0.1)
	{
		MixingImages(img1, img2, alpha, height, width, img_out);

		ImageShow((char*)"output", img_out, height, width);


	}
	return 0;
}

void Stretch1(int** img, int a, int height, int width, int** img_out)
{


	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			if (img[y][x] < a)
			{
				img_out[y][x] = ((255 / (float)a) * img[y][x] + 0.5); //반올림하기 위해 0.5를 더한다.
				//img_out[y][x] = (255.0 / a) * img[y][x]; 이렇게해도 된다.
				/*기본적으로 C에서 두 개의 정수를 나누면 그 결과도 정수가 된다.
				예를 들어, 255 / 100은 정수 나눗셈이므로 결과가 2로 나오는 반면,
				(float)a로 변환해서 255 / (float)100으로 하면 실수 나눗셈이 되어 결과는 2.55가된다. */
			}
			else {
				img_out[y][x] = 255;
			}
		}


	}

}


int ex1002_1()
{
	int height, width;
	int** img = ReadImage((char*)"./images/lena.png", &height, &width);

	int** img_out = (int**)IntAlloc2(height, width);

	int a = 100;



	Stretch1(img, a, height, width, img_out);


	ImageShow((char*)"input", img, height, width);
	ImageShow((char*)"output", img_out, height, width);


	return 0;
}


int main1() //반올림
{
	float a = 100.4;
	int b = a + 0.5;
	std::cout << a;
	ex1002_1();

	return 0;
}



void Strech2(int** img, int a,int b, int c,int d,  int height, int width, int** img_out)
{

	
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			if (img[y][x] < a)
			{
				img_out[y][x] = ((float)c / a) * img[y][x] + 0.5;
			}
			else if (img[y][x] < b)
			{
				img_out[y][x] = ((float)d - c) / (b - a) * (img[y][x] - a) + c + 0.5;
			}
			else {
				img_out[y][x] = (255.0 - d) / (255 - b) * (img[y][x] - b) + d + 0.5;
			}
		}
	}

}

struct Parameter
{
	int a, b, c, d;


}; //

void Stretch3(
	Parameter param, int** img, int height, int width, int** img_out)
{

		for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			if (img[y][x] < param.a)
			{
				img_out[y][x] = ((float)param.c / param.a) * img[y][x] + 0.5;
			}
			else if (img[y][x] < param.b)
			{
				img_out[y][x] = ((float)param.d - param.c) / (param.b - param.a) * (img[y][x] - param.a) + param.c + 0.5;
			}
			else {
				img_out[y][x] = (255.0 - param.d) / (255 - param.b) * (img[y][x] - param.b) + param.d + 0.5;
			}
		}
	}



}

struct ParameterAll {
	int a, b, c, d;
	int** img;
	int height;
	int width;
	int** img_out;
};

void Stretch4(ParameterAll p)
	{

		for (int y = 0; y < p.height; y++)
		{
			for (int x = 0; x < p.width; x++)
			{
				if (p.img[y][x] < p.a)
				{
					p.img_out[y][x] = ((float)p.c / p.a) * p.img[y][x] + 0.5;
				}
				else if (p.img[y][x] < p.b)
				{
					p.img_out[y][x] = ((float)p.d - p.c) / (p.b - p.a) * (p.img[y][x] - p.a) + p.c + 0.5;
				}
				else {
					p.img_out[y][x] = (255.0 - p.d) / (255 - p.b) * (p.img[y][x] - p.b) + p.d + 0.5;
				}
			}
		}






}


int ex1108_1(void)
{
	int height, width;
	int** img = ReadImage((char*)"./images/lena.png", &height, &width);
	int** img_out = (int**)IntAlloc2(height, width);


	//int a = 150, b = 150, c = 50, d = 200;

	//Strech2(img,  a, b, c,d, height, width, img_out);

	//Parameter param라고 해도 된다.
	//struct Parameter param;
	//param.a = 100;
	//param.b = 150;
	//param.c = 50;
	//param.d = 200;


	//Stretch3(param, img,  height, width, img_out);
	struct ParameterAll p;
	p.a = 100;
	p.b = 150;
	p.c = 50;
	p.d = 200;
	p.img = img;
	p.height = height;
	p.width = width;
	p.img_out = img_out;



	Stretch4(p);


	ImageShow((char*)"input", img, height, width);
	ImageShow((char*)"output", img_out, height, width);

	return 0;

}



int GetCount(int value, int** img, int height, int width)
{
	int count = 0;

	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			if (img[y][x] == value)
			{
				count++;
			}

		}

	}
	return count;

}


void GetHistogram(int** img, int height, int width, int* histogram)
{
	for (int value = 0; value < 256; value++) {
		histogram[value] = GetCount(value, img, height, width);
	}
}


/*히스토그램*/
int ex1008_2(void) {
	int height, width;
	int** img = ReadImage((char*)"./images/lena.png", &height, &width);
	
	int histogram[256];

	GetHistogram(img, height, width, histogram);

	

	DrawHistogram((char*)"histo", histogram);



	//ImageShow((char*)"input", img, height, width

	return 0;

}
// 히스토그램을 계산하는 함수
void GetHistogram2(int** img, int height, int width, int* histogram)
{
	for (int value = 0; value < 256; value++) {
		histogram[value] = GetCount(value, img, height, width);
	}
}

/*// 누적 히스토그램을 계산하는 함수*/
void GetChistogram(int** img, int height, int width, int* chist)
{
	int histogram[256] = { 0 };
	GetHistogram2(img, height, width, histogram);

	// 누적 히스토그램 계산
	chist[0] = histogram[0];
	for (int n = 1; n < 256; n++)
	{
		chist[n] = chist[n - 1] + histogram[n]; // n=1, n=2, ..., 255
	}
}

void HistogramEqualiztion(int** img, int height, int width, int** img_out)
{


	/*1. chist*/
	int chist[256] = { 0 };

	// 누적 히스토그램 계산
	GetChistogram(img, height, width, chist); //누적히스토그램구하기


	/*2. normalization*/
	int norm_chist[256] = { 0 };
	for (int n = 0; n < 256; n++) {

		norm_chist[n] = (float)chist[n] / (width * height) * 255 + 0.5;

	}

	//3. mapping usin 'norm_chist[]'
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			img_out[y][x] = norm_chist[img[y][x]];
		}

	}





}

int ex1008_3(void)
{
	int height, width;

	// 이미지 읽어오기
	int** img = ReadImage((char*)"./images/lena.png", &height, &width);
	int** img_out = (int**)IntAlloc2(height, width);

	HistogramEqualiztion(img, height, width, img_out);





	// 누적 히스토그램 출력
	ImageShow((char*)"input", img, height, width);
	ImageShow((char*)"output", img_out, height, width);
	
	int hist_input[256] = { 0 };
	int hist_output[256] = { 0 };
	
	GetHistogram2(img, height, width, hist_input);
	GetHistogram2(img_out, height, width, hist_output);


	DrawHistogram((char*)"input_hist", hist_input);
	DrawHistogram((char*)"output_hist", hist_output);


	return 0;
}

void MeanFilter3x3( int height,int width, int** img, int** img_out) {
	int y, x;


	for (y = 1; y < height - 1; y++) {
		for (x = 1; x < width - 1; x++) {

			img_out[y][x] = (img[y - 1][x - 1] + img[y - 1][x] + img[y - 1][x + 1]
				+ img[y][x - 1] + img[y][x] + img[y][x + 1]
				+ img[y + 1][x - 1] + img[y + 1][x] + img[y + 1][x + 1]) / 9.0 + 0.5;

		}

	}

	//위에 가로줄
	y = 0;
	for (x = 0; x < width; x++) {
		img_out[y][x] = img[y][x];
	}


	//아래 가로줄
	y = height - 1;
	for (x = 0; x < width; x++) {
		img_out[y][x] = img[y][x];
	}


	//왼쪽 세로줄
	x = 0;
	for (y = 0; y < width; y++) {
		img_out[y][x] = img[y][x];
	}


	//오른쪽 세로줄
	x = width - 1;
	for (y = 0; y < width; y++) {
		img_out[y][x] = img[y][x];
	}



}



int getMean(int y, int x, int** img) {


	/*int avg= (img[y - 1][x - 1] + img[y - 1][x] + img[y - 1][x + 1]
	+ img[y][x - 1] + img[y][x] + img[y][x + 1]
	+ img[y + 1][x - 1] + img[y + 1][x] + img[y + 1][x + 1]) / 9.0 + 0.5;*/


	int sum;
	for (int i = -1; i <=1; i++) {
		for (int j = -1; j <=1; j++) {

			sum =img[y + i][x + j];
		}


	}

	int avg =sum / 9.0 + 0.5;




	return avg;


}


/*5x5 필터*/
void MeanFilter5x5(int height, int width, int** img, int** img_out) {
	int y, x;


	for (y = 1; y < height - 1; y++) {
		for (x = 1; x < width - 1; x++) {

			img_out[y][x] = getMean(x,y,img);

		}

	}

	//위에 가로줄
	y = 0;
	for (x = 0; x < width; x++) {
		img_out[y][x] = img[y][x];
	}


	//아래 가로줄
	y = height - 1;
	for (x = 0; x < width; x++) {
		img_out[y][x] = img[y][x];
	}


	//왼쪽 세로줄
	x = 0;
	for (y = 0; y < width; y++) {
		img_out[y][x] = img[y][x];
	}


	//오른쪽 세로줄
	x = width - 1;
	for (y = 0; y < width; y++) {
		img_out[y][x] = img[y][x];
	}



}






void MeanFilter5x5_(int height, int width, int** img, int** img_out) {
	int y, x;


	for (y = 2; y < height - 2; y++) {
		for (x = 2; x < width - 2; x++) {

			
			getMean(y, x, img);
		}

	}

	//위에 가로줄

	for (y = 0; y <= 1; y++) {

		for (x = 0; x < width; x++) {
			img_out[y][x] = img[y][x];
		}
		for (x = 0; x < width; x++) {
			img_out[y][x] = img[y][x];
		}



	}
	
	for (y = height - 1; y <= height - 2; y--) {
		//아래 가로줄
		for (x = 0; x < width; x++) {
			img_out[y][x] = img[y][x];
		}
		for (x = 0; x < width; x++) {
			img_out[y][x] = img[y][x];
		}


	}


	


	//왼쪽 세로줄

	for (int x = 0; x <= 1; x++) {
		for (y = 0; y < width; y++) {
			img_out[y][x] = img[y][x];
		}
		for (y = 0; y < width; y++) {
			img_out[y][x] = img[y][x];
		}


	}
	

	//오른쪽 세로줄
	for (x = height - 1; x <= height - 2; x--) {
		for (y = 0; y < width; y++) {
			img_out[y][x] = img[y][x];
		}
		for (y = 0; y < width; y++) {
			img_out[y][x] = img[y][x];
		}
	}

}



int getMean5x5(int y, int x, int** img) {


	/*int avg= (img[y - 1][x - 1] + img[y - 1][x] + img[y - 1][x + 1]
	+ img[y][x - 1] + img[y][x] + img[y][x + 1]
	+ img[y + 1][x - 1] + img[y + 1][x] + img[y + 1][x + 1]) / 9.0 + 0.5;*/


	int sum;
	for (int i = -2; i <= 2; i++) {
		for (int j = -2; j <= 2; j++) {

			sum = img[y + i][x + j];
		}


	}

	int avg = sum / 25.0 + 0.5;




	return avg;


}



int main(void)
{
	int height, width;

	// 이미지 읽어오기
	int** img = ReadImage((char*)"./images/lena.png", &height, &width);
	int** img_out = (int**)IntAlloc2(height, width);

	
	//MeanFilter(height, width, img, img_out);
	MeanFilter5x5(height, width, img, img_out);




	ImageShow((char*)"input", img, height, width);
	ImageShow((char*)"output", img_out, height, width);
}