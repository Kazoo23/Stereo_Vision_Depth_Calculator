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

    cv::cvtColor(R, RG, cv::COLOR_BGR2GRAY);

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

    cv::Mat scaledDisparity;

    disparity165.convertTo(scaledDisparity, CV_32F, 1.0 / 16.0);

    double minDepth, maxDepth;

    double focalLength = 3740.0;
    double baseline = 0.160;

    cv::Mat disparityMask = scaledDisparity <= 30;

    cv::Mat depthMap;

    cv::divide(focalLength * baseline, scaledDisparity, depthMap);

    std::cout << "Min depth: " << minDepth << " m\n";
    std::cout << "Max depth: " << maxDepth << " m\n";

    depthMap.setTo(0, disparityMask);

    cv::minMaxLoc(depthMap, &minDepth, &maxDepth);

    std::cout << "Min depth: " << minDepth << " m\n";
    std::cout << "Max depth: " << maxDepth << " m\n";

    cv::Mat depth8U;
    cv::normalize(depthMap, depth8U, 0, 255, cv::NORM_MINMAX, CV_8U);

    cv::normalize(depthMap,depth8U, 0,255,cv::NORM_MINMAX,CV_8UC1);

    cv::Mat coloredDisparity;
    cv::applyColorMap(disparity8U, coloredDisparity, cv::COLORMAP_JET);

    cv::Mat coloredDepth;
    cv::applyColorMap(depth8U, coloredDepth, cv::COLORMAP_JET);

    coloredDepth.setTo(cv::Scalar(0, 0, 0), disparityMask);

    cv::imshow("Left Image", L);
    cv::imshow("Right Image", R);
    cv::imshow("Disparity Map", coloredDisparity);
    cv::imshow("Depth Map", coloredDepth);

    cv::waitKey(0);
    return 0;
}
