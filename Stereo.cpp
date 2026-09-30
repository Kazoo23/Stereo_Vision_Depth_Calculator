#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include <cfloat>
#include <cstdlib>

cv::Mat myStereo(const cv::Mat &left, const cv::Mat &right, int maxDisp, int block = 7, int focalLength = 3740)
{
    cv::Mat disparityMap = cv::Mat::zeros(left.size(), CV_8UC1);
    for (int x = 0; x < left.cols; x++)
    {
        for (int y = 0; y < left.rows; y++)
        {
            int bestDisparity = -1;
            double bestDifference = DBL_MAX;
            for (int d = 0; d < maxDisp; d++)
            {
                if (x - d < 0)
                {
                    continue;
                }
                double disparity = std::abs(left.at<uchar>(y, x) - right.at<uchar>(y, x - d));
                if (disparity < bestDifference)
                {
                    bestDisparity = d;
                    bestDifference = disparity;
                }
            }
            if (bestDisparity == -1)
            {
                continue;
            }
            else
            {
                disparityMap.at<uchar>(y, x) = bestDisparity;
            }
        }
    }
    return disparityMap;
}

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
    /*
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

    */

    cv::Mat disparityMap = myStereo(LG, RG, numDisparities);

    cv::Mat normalizedDisparity;

    cv::normalize(
        disparityMap,
        normalizedDisparity,
        0,
        255,
        cv::NORM_MINMAX,
        CV_8U
    );

    double minVal, maxVal;
    cv::minMaxLoc(normalizedDisparity, &minVal, &maxVal);

    std::cout << "Min disparity: " << minVal << std::endl;
    std::cout << "Max disparity: " << maxVal << std::endl;

    cv::Mat coloredDisparity;
    cv::applyColorMap(normalizedDisparity, coloredDisparity, cv::COLORMAP_JET);

    cv::imshow("Left Image", L);
    cv::imshow("Disparity Map", coloredDisparity);

    cv::waitKey(0);
    return 0;
}