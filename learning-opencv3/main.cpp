#include <opencv2/opencv.hpp>
#include <iostream>

int main(int argc, char **argv) {
    cv::Mat img = cv::imread("/Volumes/develop/code-repository/practice-code/learning-opencv3/resources/tmdyh.jpg",
                             -1);// load the image to cv::Mat data structure
    if (img.empty()) {
        return -1;
    }
    std::cout << "sss" << std::endl;
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(), 0.5, 0.5);
    cv::imwrite("/Volumes/develop/code-repository/python/picture-processing/"
                "resources/2.jpg", img);
    return 0;
}

