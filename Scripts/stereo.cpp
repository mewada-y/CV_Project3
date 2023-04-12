#include <cstdio>
#include <opencv2/opencv.hpp>
#include </usr/include/eigen3/Eigen/Dense>

using namespace cv;
using namespace std;
using Eigen::MatrixXd;
 
/*
    Author - Yash Mewada (mewada.y@northeastern.edu) & Pratik Baldota (baldota.p@northeastern.edu)
    Created - April 8, 2023
*/

/* Class for Image Mosaicing function declaration*/
class imageMosaicing
{
private:
    /* data */
public:
    string path_to_images;
    Mat img1;
    Mat img2;
    Mat soble_x = (Mat_<float>(3,3) << -1,0,1,-2,0,2,-1,0,1);
    Mat soble_y = (Mat_<float>(3,3) << -1,-2,-1,0,0,0,1,2,1);
    const int MIN_POINTS = 4;
    const double THRESHOLD = 10;
    const int MAX_ITERATIONS = 1000;
    double ncc_thres = 0.9;

    imageMosaicing(string _path);
    pair<vector<Point>, vector<Point>> perform_harris(int thresh);
    double calc_NCC(Mat temp1, Mat temp2);
    vector<pair<Point, Point>> get_correspondences(vector<Point> c1,vector<Point> c2);
    void visualise_corress(vector<pair<Point, Point>> corresspondences);
    vector<pair<Point, Point>> estimate_homography_ransac(vector<Point> src_points, vector<Point> dst_points);
    vector<Point> harris_detector_for_img1(int thres = 250);
    vector<Point> harris_detector_for_img2(int thres = 250);
    Mat findFundamentalMat(vector<pair<Point, Point>> corresspondingPts);
    Mat kron(vector<pair<Point,Point>> corees_pts);

};

imageMosaicing::imageMosaicing(string _path)
{
    cout << "This is a demo for Image Mosaicing code" << endl;
    this->path_to_images = _path; 
    img1 = imread(path_to_images + string("building_left.png"),IMREAD_GRAYSCALE);
    img2 = imread(path_to_images + string("building_right.png"),IMREAD_GRAYSCALE);
    resize(img1, img1, Size(), 0.75, 0.75);
    resize(img2, img2, Size(), 0.75, 0.75);
    // cvtColor(img1, img1, COLOR_BGR2GRAY);
    // cvtColor(img2, img2, COLOR_BGR2GRAY);
}

vector<Point> imageMosaicing::harris_detector_for_img1(int thres){
    Mat gradient_x, gx, gxy;
    Mat gradient_y, gy;
    Mat r_norm;
    Mat r = Mat::zeros(img1.size(), CV_32FC1);

    filter2D(img1,gradient_x,CV_32F,soble_x);
    filter2D(img1,gradient_y,CV_32F,soble_y);

    gx = gradient_x.mul(gradient_x);
    gy = gradient_y.mul(gradient_y);
    gxy = gradient_x.mul(gradient_y);

    GaussianBlur(gx,gx,Size(5,5),1.4);
    GaussianBlur(gy,gy,Size(5,5),1.4);
    GaussianBlur(gxy,gxy,Size(5,5),1.4);

    for(int i = 0; i < img1.rows; i++){
        for(int j = 0; j < img1.cols; j++){
            float a = gx.at<float>(i, j);
            float b = gy.at<float>(i, j);
            float c = gxy.at<float>(i, j);
            float det = a*c - b*b;
            float trace = a + c;
            r.at<float>(i,j) = det - 0.04*trace*trace;
        }
    }
    
    normalize(r, r_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat()); 
    
    Mat corners = Mat::zeros(img1.size(),CV_8UC1);
    vector<Point> corner_coor;
    Mat cr;
    cvtColor(img1,cr,COLOR_GRAY2BGR);
     for (int i = 1; i < r_norm.rows; i++) {
        for (int j = 1; j < r_norm.cols; j++) {
            // Check if current pixel is a local maximum
            if ((int) r_norm.at<float>(i, j) > thres
                && r_norm.at<float>(i, j) > r_norm.at<float>(i - 1, j - 1)
                && r_norm.at<float>(i, j) > r_norm.at<float>(i - 1, j)
                && r_norm.at<float>(i, j) > r_norm.at<float>(i - 1, j + 1)
                && r_norm.at<float>(i, j) > r_norm.at<float>(i, j - 1)
                && r_norm.at<float>(i, j) > r_norm.at<float>(i, j + 1)
                && r_norm.at<float>(i, j) > r_norm.at<float>(i + 1, j - 1)
                && r_norm.at<float>(i, j) > r_norm.at<float>(i + 1, j)
                && r_norm.at<float>(i, j) > r_norm.at<float>(i + 1, j + 1)
                ) {
                corner_coor.push_back({j,i});
                circle(cr, Point(j,i), 1, Scalar(30, 255, 30), 2,8,0);
            }
        }
    }
    cv::imshow("corners",cr);
    cv::waitKey(0);
    return corner_coor;
}

vector<Point> imageMosaicing::harris_detector_for_img2(int thres){
    Mat gradient_x, gx2, gxy;
    Mat gradient_y, gy2;
    Mat r_norm;
    Mat r = Mat::zeros(img2.size(), CV_32FC1);
    Mat corners = Mat::zeros(img2.size(),CV_8UC1);
    vector<Point> corner_coor;

    filter2D(img2,gradient_x,CV_32F,soble_x);
    filter2D(img2,gradient_y,CV_32F,soble_y);
    
    gx2 = gradient_x.mul(gradient_x);
    gy2 = gradient_y.mul(gradient_y);
    gxy = gradient_x.mul(gradient_y);
    
    GaussianBlur(gx2,gx2,Size(5,5),1.4);
    GaussianBlur(gy2,gy2,Size(5,5),1.4);
    GaussianBlur(gxy,gxy,Size(5,5),1.4);
    
    for(int i = 0; i < img2.rows; i++){
        for(int j = 0; j < img2.cols; j++){
            float a = gx2.at<float>(i, j);
            float b = gy2.at<float>(i, j);
            float c = gxy.at<float>(i, j);
            float det = a*c - b*b;
            float trace = a + c;
            r.at<float>(i,j) = det - 0.04*trace*trace;
        }
    }
    normalize(r, r_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat()); 
    
    
    Mat cr;
    cvtColor(img2,cr,COLOR_GRAY2BGR);
     for (int i = 1; i < r_norm.rows; i++) {
        for (int j = 1; j < r_norm.cols; j++) {
            // Check if current pixel is a local maximum
            if ((int) r_norm.at<float>(i, j) > thres
                && r_norm.at<float>(i, j) > r_norm.at<float>(i - 1, j - 1)
                && r_norm.at<float>(i, j) > r_norm.at<float>(i - 1, j)
                && r_norm.at<float>(i, j) > r_norm.at<float>(i - 1, j + 1)
                && r_norm.at<float>(i, j) > r_norm.at<float>(i, j - 1)
                && r_norm.at<float>(i, j) > r_norm.at<float>(i, j + 1)
                && r_norm.at<float>(i, j) > r_norm.at<float>(i + 1, j - 1)
                && r_norm.at<float>(i, j) > r_norm.at<float>(i + 1, j)
                && r_norm.at<float>(i, j) > r_norm.at<float>(i + 1, j + 1)
                ) {                
                corner_coor.push_back({j,i});
                circle(cr, Point(j,i), 1, Scalar(30, 255, 30), 2,8,0);
            }
        }
    }
    cv::imshow("corners",cr);
    cv::waitKey(0);
    return corner_coor;
}

pair<vector<Point>, vector<Point>> imageMosaicing::perform_harris(int thresh){
    Mat dst, dst_norm, dst_norm_scaled;
    Mat dst2, dst_norm2, dst_norm_scaled2;
    vector<Point> cor_1,cor_2;
    dst = Mat::zeros(img1.size(), CV_32FC1);
    dst2 = Mat::zeros(img2.size(), CV_32FC1);

    int blockSize = 2;
    int apertureSize = 5;
    double k = 0.04;

    cornerHarris(img1, dst, blockSize, apertureSize, k, BORDER_DEFAULT);
    normalize(dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat()); 
    convertScaleAbs( dst_norm, dst_norm_scaled );

    cornerHarris(img2, dst2, blockSize, apertureSize, k, BORDER_DEFAULT);
    normalize(dst2, dst_norm2, 0, 255, NORM_MINMAX, CV_32FC1, Mat()); 
    convertScaleAbs( dst_norm2, dst_norm_scaled2 );
    
    vector<Point> corner_coor;
    Mat cr1;
    cvtColor(img1,cr1,COLOR_GRAY2BGR);
    for( int i = 0; i < dst_norm.rows ; i++ )
    {
        for( int j = 0; j < dst_norm.cols; j++ )
        {
            if( (int) dst_norm.at<float>(i,j) > thresh
            && dst_norm.at<float>(i, j) > dst_norm.at<float>(i - 1, j - 1)
            && dst_norm.at<float>(i, j) > dst_norm.at<float>(i - 1, j)
            && dst_norm.at<float>(i, j) > dst_norm.at<float>(i - 1, j + 1)
            && dst_norm.at<float>(i, j) > dst_norm.at<float>(i, j - 1)
            && dst_norm.at<float>(i, j) > dst_norm.at<float>(i, j + 1)
            && dst_norm.at<float>(i, j) > dst_norm.at<float>(i + 1, j - 1)
            && dst_norm.at<float>(i, j) > dst_norm.at<float>(i + 1, j)
            && dst_norm.at<float>(i, j) > dst_norm.at<float>(i + 1, j + 1)
            )
            {
                circle(cr1, Point(j,i), 1, Scalar(30, 30, 255), 2,8,0);
                cor_1.push_back(Point(j,i));
            }
        }
    }
    Mat cr2;
    cvtColor(img2,cr2,COLOR_GRAY2BGR);
    for( int i = 0; i < dst_norm2.rows ; i++ )
    {
        for( int j = 0; j < dst_norm2.cols; j++ )
        {
            if( (int) dst_norm2.at<float>(i,j) > thresh - 10
            && dst_norm2.at<float>(i, j) > dst_norm2.at<float>(i - 1, j - 1)
            && dst_norm2.at<float>(i, j) > dst_norm2.at<float>(i - 1, j)
            && dst_norm2.at<float>(i, j) > dst_norm2.at<float>(i - 1, j + 1)
            && dst_norm2.at<float>(i, j) > dst_norm2.at<float>(i, j - 1)
            && dst_norm2.at<float>(i, j) > dst_norm2.at<float>(i, j + 1)
            && dst_norm2.at<float>(i, j) > dst_norm2.at<float>(i + 1, j - 1)
            && dst_norm2.at<float>(i, j) > dst_norm2.at<float>(i + 1, j)
            && dst_norm2.at<float>(i, j) > dst_norm2.at<float>(i + 1, j + 1)
            )
            {
                circle(cr2, Point(j,i), 1, Scalar(30, 30, 255), 2,8,0);
                cor_2.push_back(Point(j,i));
                // cout << Point(j,i) << endl;
            }
        }
    }
    Mat concated_img;
    cout << "corner1_size" << "  ";
    cout << cor_1.size() << "   ";
    cout << "corner2_size" << "  ";
    cout << cor_2.size() << endl;
    if (img1.cols != img2.cols) {
        double scale = (double)img1.cols / img2.cols;
        resize(img2, img2, Size(img1.cols, scale*img2.rows));
    }
    
    // Concatenate images vertically
    Mat result;
    cv::hconcat(cr1, cr2, concated_img);
    cv::namedWindow( "corners_window" );
    cv::imshow( "corners_window", concated_img);
    cv::imwrite( "corners_window.jpg", concated_img);
    cv::waitKey(0);
    return make_pair(cor_1,cor_2);
}

double imageMosaicing::calc_NCC(Mat temp1,Mat temp2){
    double mean1 = 0;
    for(int i=0; i<temp1.rows; i++)
    {
        for(int j=0; j<temp1.cols; j++)
        {
            mean1 += temp1.at<uchar>(i,j);
        }
    }
    mean1 = mean1/(temp1.rows*temp1.cols);
    double mean2 = 0;
    for(int i=0; i<temp2.rows; i++)
    {
        for(int j=0; j<temp2.cols; j++)
        {
            mean2 += temp2.at<uchar>(i,j);
        }
    }
    mean2 = mean2/(temp2.rows*temp2.cols);
    double std1 = 0;
    for(int i=0; i<temp1.rows; i++)
    {
        for(int j=0; j<temp1.cols; j++)
        {
            std1 += pow(temp1.at<uchar>(i,j) - mean1, 2);
        }
    }
    std1 = sqrt(std1/(temp1.rows*temp1.cols));
    double std2 = 0;
    for(int i=0; i<temp2.rows; i++)
    {
        for(int j=0; j<temp2.cols; j++)
        {
            std2 += pow(temp2.at<uchar>(i,j) - mean2, 2);
        }
    }
    std2 = sqrt(std2/(temp2.rows*temp2.cols));
    double ncc = 0;
    // int count = 0;
    for(int i=0; i<temp1.rows; i++)
    {
        for(int j=0; j<temp1.cols; j++)
        {
            ncc += (temp1.at<uchar>(i,j) - mean1)*(temp2.at<uchar>(i,j) - mean2);
            // count++;
        }
    }
    if (std1 > 0 && std2 > 0) {
        ncc = ncc/(temp1.rows*temp1.cols*std1*std2);
    } 
    else {
        ncc = 0; // or set to some other default value
    }
    // ncc = ncc/(temp1.rows*temp1.cols*std1*std2);
    return ncc;

}

vector<pair<Point, Point>> imageMosaicing::get_correspondences(vector<Point> c1,vector<Point> c2){
    Mat t1,t2;
    vector<pair<Point,Point>> corres;
    Mat temp_path,temp_path2;
    Point d = Point(0,0);

    for (int i = 0; i < c1.size() ; i++) {
        double ncc_max = 0;
        Point pt1 = c1[i];
        int p1x = pt1.x - 3;
        int p1y = pt1.y - 3;
        if (p1x < 0 || p1y < 0 || p1x + 7 >= img1.cols || p1y + 7 >= img1.rows){
            continue;
        }
        temp_path = img1(Rect(p1x, p1y, 7, 7));
        d = Point(0,0);
        int maxidx = -1;
        for (int j = 0; j < c2.size(); j++) {
            Point pt2 = c2[j];
            int p2x = pt2.x - 3;
            int p2y = pt2.y - 3;
            if (p2x < 0 || p2y < 0 || p2x + 7 >= img2.cols || p2y + 7 >= img2.rows){
                continue;
            }
            temp_path2 = img2(Rect(p2x,p2y, 7, 7));

            double temp_ncc = calc_NCC(temp_path,temp_path2);
            if (temp_ncc > ncc_max){
                ncc_max = temp_ncc;
                maxidx = j;
            }
        }
        if (c2[maxidx] != Point(0,0) && c1[i] != Point(0,0) && ncc_max > ncc_thres){
            pair<Point,Point> c;
            c.first = c1[i];
            c.second = c2[maxidx]; 
            // cout << "maxidx" << " ";
            // cout << maxidx << endl;
            corres.push_back(c);           
        }
    }
    cout << corres.size() << endl;
    Mat img_matches;
    if (img1.cols != img2.cols) {
        double scale = (double)img1.cols / img2.cols;
        resize(img2, img2, Size(img1.cols, scale*img2.rows));
    }
    
    // Concatenate images vertically
    // vconcat(img1, img2, img_matches);
    Mat cr2,cr1;
    cvtColor(img2,cr2,COLOR_GRAY2BGR);
    cvtColor(img1,cr1,COLOR_GRAY2BGR);
    hconcat(cr1, cr2, img_matches);
    for (int i = 0; i < corres.size() ; i++) {
        Point pt1 =  corres[i].first;
        Point pt2 = Point(corres[i].second.x + img1.cols, corres[i].second.y); // shift the x-coordinate of pt2 to the right of img1
        // Point pt2 = Point(fc[i].second.x, fc[i].second.y + img1.rows);
        line(img_matches, pt1, pt2, Scalar(30, 30, 255), 1);
    }
    imshow( "result_window", img_matches );
    cv::imwrite("CorrepondencespreHomography.jpg",img_matches);
    cv::waitKey(0);
    return corres;
}

void imageMosaicing::visualise_corress(vector<pair<Point, Point>> fc){
    Mat img_matches;
    if (img1.cols != img2.cols) {
        double scale = (double)img1.cols / img2.cols;
        resize(img2, img2, Size(img1.cols, scale*img2.rows));
    }
    
    // Concatenate images vertically
    // vconcat(img1, img2, img_matches);
    hconcat(img1, img2, img_matches);
    for (int i = 0; i < fc.size() ; i++) {
        Point pt1 =  fc[i].first;
        Point pt2 = Point(fc[i].second.x + img1.cols, fc[i].second.y); // shift the x-coordinate of pt2 to the right of img1
        // Point pt2 = Point(fc[i].second.x, fc[i].second.y + img1.rows);
        line(img_matches, pt1, pt2, Scalar(0, 255, 0), 1);
    }
    imshow( "result_window", img_matches );
    cv::imwrite("CorrepondencespreHomography.jpg",img_matches);
    cv::waitKey(0);
}

vector<pair<Point, Point>> imageMosaicing::estimate_homography_ransac(vector<Point> src_points, vector<Point> dst_points) {
    vector<pair<Point, Point>> bestCorrespondingPoints;
    int max_inliers = 0;
    // src_points.resize(src_points.size());
    vector<int> best_inliers;
    vector<Point> inliers1;
    vector<Point> inliers2;
    Mat best_homography = Mat::eye(3, 3, CV_64F);
    best_homography =findHomography(src_points,dst_points,RANSAC);
    vector<int> inliers;
    int num_inliers = 0;
    vector<pair<Point, Point>> temp_corres;
    int inlier_idx = -1;
    //    vector<Point2f> curr_inliers;
    for (int j = 0; j < dst_points.size(); j++) {
        Point src_point = src_points[j];
        Point dst_point = dst_points[j];
        Mat src = (Mat_<double>(3, 1) << src_point.x, src_point.y, 1);
        Mat dst = (Mat_<double>(3, 1) << dst_point.x, dst_point.y, 1);
        Mat pred_dst = best_homography * src;
        pred_dst /= pred_dst.at<double>(2, 0);
        
        double distance = norm(pred_dst-dst);
        // cout << src <<" << norm , manual >> ";SSSSS
        // cout << p << endl;
        // cout << distance << endl;
        if (distance < 1) {
            // cout << "got" << endl;
            num_inliers++;
            // inlier_idx = j;
            pair<Point,Point> c;
            c.first = src_point;
            c.second = dst_point; 
            temp_corres.push_back(c);
            inliers.push_back(j);
            // cout << num_inliers << endl;
            // curr_inliers.push_back(src_points[j]);
        }
        if (num_inliers >= max_inliers) {
            // cout << "blahh" << endl;
            max_inliers = num_inliers;
            best_inliers = inliers;
            // best_inliers = curr_inliers;
            best_homography = best_homography;
            bestCorrespondingPoints = temp_corres;
            // best_homography = estimateAffinePartial2D(src_points, dst_points, inliers, RANSAC, THRESHOLD);
        }
    }
    // cout << "max_inliers" << "    ";
    // cout << max_inliers << endl;
    vector<Point> inlier_src_points;
    vector<Point> inlier_dst_points;
    for (int i = 0; i < best_inliers.size(); i++) {
        int idx = best_inliers[i];
        inlier_src_points.push_back(src_points[idx]);
        inlier_dst_points.push_back(dst_points[idx]);
        // cout << idx << endl;
    }
    // cout << best_homography << endl;
    best_homography = findHomography(inlier_src_points,inlier_dst_points,RANSAC);
    Mat img_matches;
    if (img1.cols != img2.cols) {
        double scale = (double)img1.cols / img2.cols;
        resize(img2, img2, Size(img1.cols, scale*img2.rows));
    }
    
    // Concatenate images vertically
    // vconcat(img1, img2, img_matches);
    Mat cr2,cr1;
    cvtColor(img2,cr2,COLOR_GRAY2BGR);
    cvtColor(img1,cr1,COLOR_GRAY2BGR);
    hconcat(cr1, cr2, img_matches);
    for (int i = 0; i < bestCorrespondingPoints.size() ; i++) {
        Point pt1 =  bestCorrespondingPoints[i].first;
        Point pt2 = Point(bestCorrespondingPoints[i].second.x + img1.cols, bestCorrespondingPoints[i].second.y); // shift the x-coordinate of pt2 to the right of img1
        // Point pt2 = Point(fc[i].second.x, fc[i].second.y + img1.rows);
        line(img_matches, pt1, pt2, Scalar(30, 30, 255), 1);
    }
    imshow( "result_window", img_matches );
    // cv::imwrite("CorrepondencespostHomography.jpg",img_matches);
    cv::waitKey(0);
    // visualise_corress(bestCorrespondingPoints);
    
    return bestCorrespondingPoints;
}
Mat imageMosaicing::kron(vector<pair<Point, Point>> bestCorrespondingPoints){
    // Point xl[8][1] = {Point(0,0)};
    // Point xr[8][1] = {Point(0,0)};
    Mat kron(bestCorrespondingPoints.size(),9,CV_32F);
    cout<<bestCorrespondingPoints.size()<<endl;
    cout<<kron.size()<<endl;
    // vector<Point> xl,xr;
    vector<Point> xl,xr;
    
    for (int i = 0; i < bestCorrespondingPoints.size(); i++) {
        xl.push_back(bestCorrespondingPoints[i].first);
        xr.push_back(bestCorrespondingPoints[i].second);
    }
    Scalar xlmean,xlstd;
    Scalar xrmean,xrstd;

    meanStdDev(xl,xlmean,xlstd);
    meanStdDev(xr,xrmean,xrstd);

    // cout << xlstd << endl;
    for (int i = 0; i < bestCorrespondingPoints.size() ; i++) {
        Point pt1 =  bestCorrespondingPoints[i].first;
        Point pt2 = bestCorrespondingPoints[i].second;
        Mat row = (Mat_<float>(9,1) << pt1.x*pt2.x, pt1.x*pt2.y,pt1.x,pt1.y*pt2.x,pt1.y*pt2.y, pt1.y, pt2.x,pt2.y,1);

        kron.row(i) = row;
        // cout << row << endl;
    }
    // for (int i = 0; i < kron.rows ; i++){
    //     cout << "[";
    //     for (int j = 0; j < kron.cols; j++){
    //         cout << kron.at<float>(i,j) << " ";
    //     }
    //     cout << " ]"<< endl;
    // }
}
// Mat imageMosaicing::findFundamentalMat(vector<pair<Point, Point>> corresspondingPts){

// }

int main(){
    string path = "/home/yash/Documents/Computer_VIsion/CV_Project3/Inputs/";
        
    imageMosaicing p3(path);
    vector<Point> cor_img1,cor_img2;
    
    cor_img1 = p3.harris_detector_for_img1(230);
    cor_img2 = p3.harris_detector_for_img2(230);

    
    vector<pair<Point,Point>> corres;
    corres = p3.get_correspondences(cor_img1,cor_img2);
    cout << "done" << endl;
    
    vector<Point> src,dst;
    vector<DMatch> matches;
    for (int i = 0; i < corres.size(); i++) {
        src.push_back(corres[i].first);
        dst.push_back(corres[i].second);
    }
    vector<pair<Point,Point>> inliers;
    inliers = p3.estimate_homography_ransac(src,dst);
    p3.kron(inliers);
}