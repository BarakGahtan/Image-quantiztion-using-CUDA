///////////////////////////////////////////////// DO NOT CHANGE ///////////////////////////////////////

#include "ex1.h"

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;

int main(int argc,char **argv) 
{ 
    char *image_name = argv[1];

    Mat image = imread(image_name, 1);

    if( argc != 2 || !image.data )
    {
	printf( " No image data \n " );
	return 1;
    }

    Mat gray_image;
    cvtColor(image, gray_image, CV_BGR2GRAY);
    imwrite("grayscale.png", gray_image);

    Size sz = gray_image.size();

    assert(gray_image.isContinuous());

    uchar *image_out = new uchar[sz.width * sz.height];
    cpu_process(gray_image.data, image_out, sz.width, sz.height);

    Mat result(sz, CV_8UC1, image_out);
    imwrite("result.png", result);

    namedWindow(image_name, CV_WINDOW_AUTOSIZE);
    namedWindow("Gray image", CV_WINDOW_AUTOSIZE);
    namedWindow("Result", CV_WINDOW_AUTOSIZE);

    imshow(image_name, image);
    imshow("Gray image", gray_image);
    imshow("Result", result);

    waitKey(0);

    return 0;
}
