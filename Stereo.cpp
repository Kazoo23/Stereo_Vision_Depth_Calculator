#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>

int main(int argc, char **argv)
{

    if (argc < 2)
    {
        std::cout << "Program requires 2 images";
    }

    cv::Mat L, R;

    if (argc >= 3)
    {
        L = cv::imread(argv[1]);
        R = cv::imread(argv[2]);
    }

    if (L.empty() || R.empty())
    {
        std::cerr << "could not load images (check the paths)\n";
        return 1;
    }

    cv::Mat LG, RG;

    cv::cvtColor(L, LG, cv::COLOR_BGR2GRAY);

cv:
    cvtColor(R, RG, cv::COLOR_BGR2GRAY);

    int minDisparity = 0;
    int numDisparities = 64;
    int blockSize = 3;

    int P1 = 8 * 1 * blockSize * blockSize;
    int P2 = 32 * 1 * blockSize * blockSize;

    cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(
        minDisparity,
        numDisparities,
        blockSize,
        P1,
        P2,
        1,
        63,
        15,
        100,
        1,
        cv::StereoSGBM::MODE_SGBM_3WAY);

    cv::Mat disparity165;
    sgbm->compute(LG, RG, disparity165);

    cv::Mat disparity8U;
    disparity165.convertTo(disparity8U, CV_8U, 255.0 / (numDisparities * 16.0));

    cv::Mat coloredDisparity;
    cv::applyColorMap(disparity8U, coloredDisparity, cv::COLORMAP_JET);

    cv::namedWindow("Display Window", cv::WINDOW_AUTOSIZE);
    cv::imshow("Left Image", L);
    cv::imshow("Disparity Map", coloredDisparity);

    cv::waitKey(0);
    return 0;
}
