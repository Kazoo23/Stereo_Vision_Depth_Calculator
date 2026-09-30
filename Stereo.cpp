#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include <cfloat>
#include <cstdlib>
#include <chrono>

int compareBlock(const cv::Mat &left, const cv::Mat &right, int d, int x, int y, int block = 7)
{
    int cost = 0;
    int radius = block / 2;
    for (int x_offset = radius * -1; x_offset <= radius; x_offset++)
    {
        for (int y_offset = radius * -1; y_offset <= radius; y_offset++)
        {
            if (y_offset + y < 0 || y_offset + y >= left.rows)
            {
                continue;
            }
            if (x + x_offset < 0 || x + x_offset >= left.cols)
            {
                continue;
            }
            if (x + x_offset - d < 0 || x + x_offset - d >= left.cols)
            {
                continue;
            }
            int disparity = std::abs(left.at<uchar>(y + y_offset, x + x_offset) - right.at<uchar>(y + y_offset, x + x_offset - d));
            cost += disparity;
        }
    }
    return cost;
}

cv::Mat myStereo(const cv::Mat &left, const cv::Mat &right, int maxDisp, int block = 7, int focalLength = 3740)
{
    int radius = block / 2;
    cv::Mat disparityMap = cv::Mat::zeros(left.size(), CV_8UC1);
    for (int x = radius; x < left.cols - radius; x++)
    {
        for (int y = radius; y < left.rows - radius; y++)
        {
            int bestDisparity = -1;
            double bestCost = DBL_MAX;
            for (int d = 0; d < maxDisp; d++)
            {
                int cost = compareBlock(left, right, d, x, y, block);
                if (cost < bestCost)
                {
                    bestCost = cost;
                    bestDisparity = d;
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

    auto start = std::chrono::high_resolution_clock::now();

    cv::Mat disparityMap = myStereo(LG, RG, numDisparities);

    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        end - start);

    std::cout << "Stereo time: " << duration.count() << " ms\n";

    cv::Mat normalizedDisparity;

    cv::normalize(
        disparityMap,
        normalizedDisparity,
        0,
        255,
        cv::NORM_MINMAX,
        CV_8U);

    double minVal, maxVal;
    cv::minMaxLoc(normalizedDisparity, &minVal, &maxVal);

    std::cout << "Min disparity: " << minVal << std::endl;
    std::cout << "Max disparity: " << maxVal << std::endl;

    cv::Mat coloredDisparity;
    cv::applyColorMap(normalizedDisparity, coloredDisparity, cv::COLORMAP_JET);

    cv::imshow("Left Image", L);
    cv::imshow("Right Image", R);
    cv::imshow("Disparity Map", coloredDisparity);

    cv::waitKey(0);
    return 0;
}