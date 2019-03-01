#include <opencv2/opencv.hpp>

int main(int argc, char **argv) {
    cv::Mat img = cv::imread(argv[1], -1);// load the image to cv::Mat data structure
    if (img.empty()) {
        return -1;
    }
    cv::namedWindow("Example1",
                    cv::WINDOW_AUTOSIZE);//open a window on the screen that can contain and display an image
    cv::imshow("Example1", img);
    cv::waitKey(0);
    cv::destroyWindow("Example1");
    return 0;
}