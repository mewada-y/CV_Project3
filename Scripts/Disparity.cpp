#include <opencv2/opencv.hpp>

using namespace cv;

int main()
{
// Load left and right images
Mat imgL = imread("/home/pratik/Downloads/image1.jpeg", IMREAD_GRAYSCALE);
Mat imgR = imread("/home/pratik/Downloads/image2.jpeg", IMREAD_GRAYSCALE);

// Compute key points and descriptors for left and right images
Ptr<ORB> orb = ORB::create();
std::vector<KeyPoint> keypointsL, keypointsR;
Mat descriptorsL, descriptorsR;
orb->detectAndCompute(imgL, Mat(), keypointsL, descriptorsL);
orb->detectAndCompute(imgR, Mat(), keypointsR, descriptorsR);

// Match key points using descriptor-based matching
Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
std::vector<DMatch> matches;
matcher->match(descriptorsL, descriptorsR, matches);

// Extract matched key points
std::vector<Point2f> pointsL, pointsR;
for (int i = 0; i < matches.size(); i++) {
pointsL.push_back(keypointsL[matches[i].queryIdx].pt);
pointsR.push_back(keypointsR[matches[i].trainIdx].pt);
}

// Compute fundamental matrix
Mat F = findFundamentalMat(pointsL, pointsR, FM_RANSAC);

// Rectify images
Mat H1, H2;
stereoRectifyUncalibrated(pointsL, pointsR, F, imgL.size(), H1, H2);
Mat imgRectL, imgRectR;
warpPerspective(imgL, imgRectL, H1, imgL.size());
warpPerspective(imgR, imgRectR, H2, imgR.size());

// Compute dense disparity map
int maxDisparity = 64;
int blockSize = 5;
Ptr<StereoSGBM> stereo = StereoSGBM::create(0, maxDisparity, blockSize);
Mat disp;
stereo->compute(imgRectL, imgRectR, disp);

// Compute horizontal and vertical disparity components
Mat dispX, dispY;
Sobel(disp, dispX, CV_32F, 1, 0);
Sobel(disp, dispY, CV_32F, 0, 1);

// Compute disparity vector using color
Mat dispVec(disp.size(), CV_32FC3);
for (int y = 0; y < disp.rows; y++) {
for (int x = 0; x < disp.cols; x++) {
float dx = dispX.at<float>(y, x);
float dy = dispY.at<float>(y, x);
float mag = sqrt(dx*dx + dy*dy);
float angle = atan2(dy, dx);
if (mag > 0) {
angle = (angle + CV_PI) / (2*CV_PI);
mag = std::min(mag / 32.0f, 1.0f);
dispVec.at<Vec3f>(y, x) = Vec3f(angle, mag, mag);
} else {
dispVec.at<Vec3f>(y, x) = Vec3f(0, 0, 0);
}
}
}

// Display result
imshow("Vertical disparity component", dispY);
cv::imwrite( "Vertical Disparity.jpg", dispY );
imshow("Horizontal disparity component", dispX);
cv::imwrite( "Horizontal Disparity.jpg", dispX );
imshow("Disparity vector", dispVec);
cv::imwrite( "DisparityV.jpg", dispVec);


waitKey(0);
return 0;
} 

