#include <cstdio>
#include <opencv2/opencv.hpp>
#include </usr/include/eigen3/Eigen/Dense>
#include <random>
#include <cmath>


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
    Mat* lines1 = nullptr;
    Mat* lines2 = nullptr;
public:
    string path_to_images;
    Mat img1;
    Mat img2;
    Mat soble_x = (Mat_<float>(3,3) << -1,0,1,-2,0,2,-1,0,1);
    Mat soble_y = (Mat_<float>(3,3) << -1,-2,-1,0,0,0,1,2,1);
    const int MIN_POINTS = 4;
    const double THRESHOLD = 10;
    const int MAX_ITERATIONS = 3000;
    double ncc_thres = 0.9;
    const int SAMPLE_SIZE = 8;
    const int ERROR_THRES = 1;

    imageMosaicing(string _path);
    double calc_NCC(Mat temp1, Mat temp2);
    vector<pair<Point, Point>> get_correspondences(vector<Point> c1,vector<Point> c2);
    vector<pair<Point, Point>> estimate_homography_ransac(vector<Point> src_points, vector<Point> dst_points);
    vector<Point> harris_detector_for_img1(int thres = 250);
    vector<Point> harris_detector_for_img2(int thres = 250);
    Mat estimateFunMat(Mat A);
    Mat estimateFunndamentalRANSAC(vector<pair<Point,Point>> corees_pts);
    Mat kron(vector<pair<Point,Point>> corees_pts);
    void findEpipolarlines(vector<Point2f> inlierxl,vector<Point2f> inlierxr, Mat F, Mat imgRectR, Mat imgRectL,Mat* lines1,Mat* lines2);

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
    //imwrite("cornerimg1.jpg",cr);
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
    //imwrite("cornerimg2.jpg",cr);
    cv::waitKey(0);
    return corner_coor;
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
    RNG rng(12345);
    for (int i = 0; i < corres.size() ; i++) {
        Point pt1 =  corres[i].first;
        Point pt2 = Point(corres[i].second.x + img1.cols, corres[i].second.y); // shift the x-coordinate of pt2 to the right of img1
        // Point pt2 = Point(fc[i].second.x, fc[i].second.y + img1.rows);
        Scalar color = Scalar(rng.uniform(0,255), rng.uniform(0, 255), rng.uniform(0, 255));
        line(img_matches, pt1, pt2, color, 1);
    }
    imshow( "result_window", img_matches);
    //imwrite("CorrepondencespreRansac.jpg",img_matches);
    cv::waitKey(0);
    return corres;
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
    double mean_dist = 0;
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
        cout << distance << endl;
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
    cout << "best_homography" << endl;
    cout << best_homography << endl;
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
    RNG rng(12345);
    hconcat(cr1, cr2, img_matches);
    for (int i = 0; i < bestCorrespondingPoints.size() ; i++) {
        Point pt1 =  bestCorrespondingPoints[i].first;
        Point pt2 = Point(bestCorrespondingPoints[i].second.x + img1.cols, bestCorrespondingPoints[i].second.y); // shift the x-coordinate of pt2 to the right of img1
        // Point pt2 = Point(fc[i].second.x, fc[i].second.y + img1.rows);
        Scalar color = Scalar(rng.uniform(0,255), rng.uniform(0, 255), rng.uniform(0, 255));
        circle(img_matches, Point(pt1.x,pt1.y), 1, Scalar(30, 255, 30), 2,8,0);
        circle(img_matches, Point(pt2.x,pt2.y), 1, Scalar(30, 255, 30), 2,8,0);
        line(img_matches, pt1, pt2, color, 1);
    }
    imshow( "result_window", img_matches );
    //imwrite("CorrepondencespostRansac.jpg",img_matches);
    cv::waitKey(0);
    // visualise_corress(bestCorrespondingPoints);
    
    return bestCorrespondingPoints;
}

Mat imageMosaicing::kron(vector<pair<Point, Point>> bestCorrespondingPoints){
    Mat kron(bestCorrespondingPoints.size(),9,CV_64F);
    // cout<<bestCorrespondingPoints.size()<<endl;
    // cout<<kron.size()<<endl;
    for (int i = 0; i < bestCorrespondingPoints.size() ; i++) {
        Point2d pt1 =  bestCorrespondingPoints[i].first;
        Point2d pt2 = bestCorrespondingPoints[i].second;
        // cout<<pt1.x*pt2.x<<endl;
        Mat row = (Mat_<double>(9,1) << pt1.x*pt2.x, pt1.x*pt2.y,pt1.x,pt1.y*pt2.x,pt1.y*pt2.y, pt1.y, pt2.x,pt2.y,1);
        kron.at<double>(i,0) = row.at<double>(0,0);
        kron.at<double>(i,1) = row.at<double>(0,1);
        kron.at<double>(i,2) = row.at<double>(0,2);
        kron.at<double>(i,3) = row.at<double>(0,3);
        kron.at<double>(i,4) = row.at<double>(0,4);
        kron.at<double>(i,5) = row.at<double>(0,5);
        kron.at<double>(i,6) = row.at<double>(0,6);
        kron.at<double>(i,7) = row.at<double>(0,7);
        kron.at<double>(i,8) = row.at<double>(0,8);
        // Mat kron_row = kron.row(i).setTo(row);
        // row.copyTo(kron_row);
        //  = row;
        // cout << row << endl;
        // cout << kron.row(1) << endl;
    }
    // cout << "[";
    // for (int i = 0; i < kron.rows ; i++){
    //     for (int j = 0; j < kron.cols; j++){
    //         cout << kron.at<double>(i,j) << " ";
    //     }
    //     cout << endl;
    // }
    // cout << " ]"<< endl;
    return kron;
}

Mat imageMosaicing::estimateFunMat(Mat A){
    SVD svdTemp(A,SVD::FULL_UV);

    Mat fUtil = svdTemp.vt.row(8);
    Mat Ftemp(3,3,CV_64FC1);

    Ftemp.at<double>(0,0) = fUtil.at<double>(0,0);
    Ftemp.at<double>(0,1) = fUtil.at<double>(0,1);
    Ftemp.at<double>(0,2) = fUtil.at<double>(0,2);
    Ftemp.at<double>(1,0) = fUtil.at<double>(0,3);
    Ftemp.at<double>(1,1) = fUtil.at<double>(0,4);
    Ftemp.at<double>(1,2) = fUtil.at<double>(0,5);
    Ftemp.at<double>(2,0) = fUtil.at<double>(0,6);
    Ftemp.at<double>(2,1) = fUtil.at<double>(0,7);
    Ftemp.at<double>(2,2) = fUtil.at<double>(0,8);
    // cout << "Old F = [";
    // for (int i = 0; i < Ftemp.rows ; i++){
    //     for (int j = 0; j < Ftemp.cols; j++){
    //         cout << Ftemp.at<double>(i,j) << " ";
    //     }
    //     cout << endl;
    // }
    // cout << " ]"<< endl;
    Mat F(3,3,Ftemp.type());
    SVD svd(Ftemp,SVD::FULL_UV);
    // cout << svd.u.size() << endl;
    // cout << svd.vt.t().size() << endl;
    // cout << svd.w.size() << endl;
    svd.w.at<double>(0,2) = 0;
    Mat w(3,3,CV_64F);
    w.at<double>(0,0) = svd.w.at<double>(0,0);
    w.at<double>(0,1) = 0;
    w.at<double>(0,2) = 0;
    w.at<double>(1,0) = 0;
    w.at<double>(1,1) = svd.w.at<double>(0,1);
    w.at<double>(1,2) = 0;
    w.at<double>(2,0) = 0;
    w.at<double>(2,1) = 0;
    w.at<double>(2,2) = 0;
    F = svd.u*w*svd.vt;
    // cout << "New F = [";
    // for (int i = 0; i < Ftemp.rows ; i++){
    //     for (int j = 0; j < Ftemp.cols; j++){
    //         cout << Ftemp.at<double>(i,j) << " ";
    //     }
    //     cout << endl;
    // }
    // cout << " ]"<< endl;
    Ftemp.at<double>(3,3) = 1;

    return Ftemp;
    
    
}

Mat imageMosaicing::estimateFunndamentalRANSAC(vector<pair<Point,Point>> corees_pts){
    Mat bestF;
    
    int bestInliercnt = 0;
    vector<Point> xl,xr;
    for (int i = 0; i < corees_pts.size(); i++) {
        xl.push_back(corees_pts[i].first);
        xr.push_back(corees_pts[i].second);
    }
    // Get 8 Random points
    for(int i=0;i<MAX_ITERATIONS;i++){
        vector<Point2f> sampxl,sampxr;
        random_shuffle(corees_pts.begin(),corees_pts.end());
        for(int j =0;j<SAMPLE_SIZE;j++){
            sampxl.push_back(xl[j]);
            sampxr.push_back(xr[j]);
        }
        Mat F = findFundamentalMat(sampxl,sampxr,FM_8POINT);
        int inlierCount = 0;
        for (int k = 0; k < xl.size(); k++) {
            // Mat x1(xl[i].);
            // Mat x2(xr);
            double error = abs(xr[k].x * F.at<double>(0,0) * xl[k].x +
                               xr[k].x * F.at<double>(0,1) * xl[k].y +
                               F.at<double>(0,2) * xl[k].x +
                               xr[k].y * F.at<double>(1,0) * xl[k].x +
                               xr[k].y * F.at<double>(1,1) * xl[k].y +
                               F.at<double>(1,2) * xl[k].y +
                               F.at<double>(2,0) * xl[k].x +
                               F.at<double>(2,1) * xl[k].y +
                               F.at<double>(2,2));
            // cout << error << endl;
            if (error < ERROR_THRES){
                inlierCount++;
                // cout << "error < 1" << endl;
            }
        }
        if (inlierCount > bestInliercnt){
            bestInliercnt = inlierCount;
            bestF = F;
        }
    }
    vector<Point> inlierxL,inlierxR;
    for (int k = 0; k < xl.size(); k++) {
            // Mat x1(xl[i].);
            // Mat x2(xr);
            double error = abs(xr[k].x * bestF.at<double>(0,0) * xl[k].x +
                               xr[k].x * bestF.at<double>(0,1) * xl[k].y +
                               bestF.at<double>(0,2) * xl[k].x +
                               xr[k].y * bestF.at<double>(1,0) * xl[k].x +
                               xr[k].y * bestF.at<double>(1,1) * xl[k].y +
                               bestF.at<double>(1,2) * xl[k].y +
                               bestF.at<double>(2,0) * xl[k].x +
                               bestF.at<double>(2,1) * xl[k].y +
                               bestF.at<double>(2,2));
            if (error < ERROR_THRES){
                inlierxL.push_back(xl[k]);
                inlierxR.push_back(xr[k]);
                // cout << "error < 1" << endl;
            }
        }
    bestF = findFundamentalMat(inlierxL,inlierxR,FM_8POINT);
    return bestF;

}

void imageMosaicing::findEpipolarlines(vector<Point2f> inlierxl,vector<Point2f> inlierxr, Mat F, Mat imgRectR, Mat imgRectL,Mat* lines1,Mat* lines2){
    // Mat lines1;
    Mat color1;
    cvtColor(imgRectR,color1,COLOR_GRAY2BGR);
    computeCorrespondEpilines(cv::Mat(inlierxl),1,F,*lines1);
    RNG rng(12345);
    for (int i = 0; i < lines1->rows ; i++){
        for (int j = 0; j < lines1->cols; j++){
            Scalar color = Scalar(rng.uniform(0,255), rng.uniform(0, 255), rng.uniform(0, 255));
            line(color1,Point(0,-(lines1->at<Vec3f>(i,j))[2]/ lines1->at<Vec3f>(i,j)[1]),Point(imgRectR.cols,-(lines1->at<Vec3f>(i,j)[2] + lines1->at<Vec3f>(i,j)[0] * imgRectR.cols)/ lines1->at<Vec3f>(i,j)[1]),color);
        }
    }


    // Mat lines2;
    Mat color2;
    cvtColor(imgRectL,color2,COLOR_GRAY2BGR);
    computeCorrespondEpilines(cv::Mat(inlierxr),2,F,*lines2);
    for (int i = 0; i < lines2->rows ; i++){
        for (int j = 0; j < lines2->cols; j++){
            Scalar color = Scalar(rng.uniform(0,255), rng.uniform(0, 255), rng.uniform(0, 255));
            line(color2,
                Point(0,-(lines2->at<Vec3f>(i,j))[2]/ lines2->at<Vec3f>(i,j)[1]),
                Point(imgRectL.cols,-(lines2->at<Vec3f>(i,j)[2] + lines2->at<Vec3f>(i,j)[0] * imgRectL.cols)/ lines2->at<Vec3f>(i,j)[1]),color);
        }
    }
    Mat epipolarLines;
    hconcat(color1,color2,epipolarLines);
    imshow("Epipolarlines",epipolarLines);
    //imwrite("Epipolarlines.jpg",epipolarLines);
    waitKey(0);

    // return lines1,lines2;
}



int main(){
    string path = "/home/yash/Documents/Computer_VIsion/CV_Project3/CV_Project3/Inputs/";
    Mat imgL = imread(path + string("building_left.png"),IMREAD_GRAYSCALE);
    Mat imgR = imread(path + string("building_right.png"),IMREAD_GRAYSCALE);

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
    Mat A = p3.kron(inliers);
    Mat svdF = p3.estimateFunMat(A);
    
    Mat F = p3.estimateFunndamentalRANSAC(inliers);
    cout << "Ransac F = [";
    for (int i = 0; i < F.rows ; i++){
        for (int j = 0; j < F.cols; j++){
            cout << F.at<double>(i,j) << " ";
        }
        cout << endl;
    }
    cout << " ]"<< endl;

    cout << "SVD F = [";
    for (int i = 0; i < svdF.rows ; i++){
        for (int j = 0; j < svdF.cols; j++){
            cout << svdF.at<double>(i,j) << " ";
        }
        cout << endl;
    }
    cout << " ]"<< endl;


    vector<Point2f> inlierxl,inlierxr;
    for (int i = 0; i < inliers.size(); i++) {
        inlierxl.push_back(inliers[i].first);
        inlierxr.push_back(inliers[i].second);
    }
    // Rectify the images
    Mat H1, H2;
    stereoRectifyUncalibrated(inlierxl,inlierxr,F,imgL.size(),H1,H2);
    Mat imgRectL, imgRectR,rectimgs;
    warpPerspective(imgL, imgRectL, H1, imgL.size());
    warpPerspective(imgR, imgRectR, H2, imgR.size());
    hconcat(imgRectL,imgRectR,rectimgs);
    imshow("rectL",rectimgs);
    //imwrite("Rectified Images.jpg",rectimgs);
    waitKey(0);
    Mat epi1,epi2;
    p3.findEpipolarlines(inlierxl,inlierxr,F,imgR,imgL,&epi1,&epi2);
    
    Ptr<StereoSGBM> stereo = StereoSGBM::create(0, 64, 20);
    // stereo->setMode(StereoSGBM::MODE_SGBM_3WAY);
    // Compute disparity map
    Mat disp; Mat dispor;
    cv::Mat disp_x, disp_y; // horizontal and vertical disparity maps
    stereo->compute(imgRectL, imgRectR, disp);
    imshow("Disparity Vector Map", disp);
    normalize(disp, disp, 0, 255, NORM_MINMAX, CV_8U);
    imshow("Disparity", disp);

    Mat vertDisp = Mat::zeros(disp.size(), CV_32FC1);
    for (int y = 1; y < disp.rows; y++)
    {
        for (int x = 0; x < disp.cols; x++)
        {
            float dispDiff = disp.at<short>(y, x) - disp.at<short>(y-1, x);
            vertDisp.at<float>(y, x) = dispDiff;
        }
    }

    // Output the vertical disparity as an image
    normalize(vertDisp, vertDisp, 0, 255, NORM_MINMAX, CV_8UC1);
    imwrite("vertical_disparity.png", vertDisp);
    waitKey(0);

    Mat dispVec = Mat::zeros(disp.size(), CV_32FC3);
    for (int y = 0; y < disp.rows; y++) {
        for (int x = 0; x < disp.cols; x++) {
        float dx = disp.at<float>(y, x);
        float dy = vertDisp.at<float>(y, x);
        float mag = sqrt(dx*dx + dy*dy);
        float angle = atan2(dy, dx);
        if (angle<0) angle += CV_PI;
        angle *= 180 / CV_PI;
        mag = std::min(mag, 32.0f);
        mag /= 32.0f;
        dispVec.at<Vec2f>(y, x) = Vec2f(angle, mag,mag);
        }
    }
    imshow("Disparity vector", dispVec);
    normalize(dispVec, dispVec, 0, 255, NORM_MINMAX, CV_8U);
    imwrite("Disparity_vector.png",dispVec);
    waitKey(0);
    // cv::waitKey(0);

    // Ptr<StereoSGBM> stereo1 = StereoSGBM::create(0, 60, 20);
    // stereo1->setMode(StereoSGBM::MODE_SGBM);
    // // Compute disparity map
    // Mat disp1; 
    // stereo1->compute(imgRectL, imgRectR, disp1);
    // imshow("Disparity Vector Map", disp1);
    // normalize(disp1, disp1, 0, 255, NORM_MINMAX, CV_8U);
    // imshow("Disparity1", disp1);
    // cv::waitKey(0);
}