
#include <iostream>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;


// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        // pixel coordinates
        pt.x = Y.at<double>(0, 0) / Y.at<double>(2, 0); 
        pt.y = Y.at<double>(1, 0) / Y.at<double>(2, 0); 

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        { 
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // eof loop over all Lidar points
}

/* 
* The show3DObjects() function below can handle different output image sizes, but the text output has been manually tuned to fit the 2000x2000 size. 
* However, you can make this function work for other sizes too.
* For instance, to use a 1000x1000 size, adjusting the text positions by dividing them by 2.
*/
void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top=1e8, left=1e8, bottom=0.0, right=0.0; 
        float xwmin=1e8, ywmin=1e8, ywmax=-1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin<xw ? xwmin : xw;
            ywmin = ywmin<yw ? ywmin : yw;
            ywmax = ywmax>yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top<y ? top : y;
            left = left<x ? left : x;
            bottom = bottom>y ? bottom : y;
            right = right>x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),cv::Scalar(0,0,0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left, bottom+50), cv::FONT_ITALIC, 1, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax-ywmin);
        putText(topviewImg, str2, cv::Point2f(left, bottom+125), cv::FONT_ITALIC, 1, currColor);  
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    cv::namedWindow(windowName, 1);
    cv::imshow(windowName, topviewImg);

    if(bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}


// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    vector<float> matchedDistances;
    float tempDistance;
    for (auto match : kptMatches)
    {
        const cv::KeyPoint kptPrev = kptsPrev[match.queryIdx];
        const cv::KeyPoint kptCurr = kptsCurr[match.trainIdx];
        
        tempDistance = sqrt(pow(kptPrev.pt.x-kptCurr.pt.x, 2) + pow(kptPrev.pt.y-kptCurr.pt.y, 2));
        matchedDistances.push_back(tempDistance);
        //cout << "match.distance: " <<match.distance << endl;
        //cout << "tempDistance: "<<tempDistance << endl;
        
    }

    sort(matchedDistances.begin(), matchedDistances.end());
    
    /*
    for (int i = 0; i < matchedDistances.size() ; i++)
    {
        cout << "matchedDistance[" <<i << "]: " <<matchedDistances[i] << endl;
    }
    cout << "kptMatches.size(): "<< kptMatches.size() << endl;
    */
    const float Q1 = matchedDistances[matchedDistances.size() / 4];
    const float Q3 = matchedDistances[matchedDistances.size() * 3/ 4];
    const float IQR = Q3 - Q1;
    const float lowOutlier = Q1 - 1.5 * IQR;
    const float highOutlier = Q3 + 1.5 * IQR;


    for (auto match : kptMatches)
    {
        const cv::KeyPoint kptPrev = kptsPrev[match.queryIdx];
        const cv::KeyPoint kptCurr = kptsCurr[match.trainIdx];
        
        tempDistance = sqrt(pow(kptPrev.pt.x-kptCurr.pt.x, 2) + pow(kptPrev.pt.y-kptCurr.pt.y, 2));

        if (boundingBox.roi.contains(kptCurr.pt) && lowOutlier < tempDistance && tempDistance < highOutlier )
        {
            boundingBox.keypoints.push_back(kptCurr);
            boundingBox.kptMatches.push_back(match);
        }
    }
}



// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    
    vector<double> distRatios;

    for (int i = 0; i < kptMatches.size(); i++)
    {
        const cv::KeyPoint originPrev = kptsPrev[kptMatches[i].queryIdx];
        const cv::KeyPoint originCurr = kptsCurr[kptMatches[i].trainIdx];

        for (int j = i; j < kptMatches.size(); j++)
        {
            const cv::KeyPoint kptPrev = kptsPrev[kptMatches[j].queryIdx];
            const cv::KeyPoint kptCurr = kptsCurr[kptMatches[j].trainIdx];
            const double distPrev = sqrt(pow(kptPrev.pt.x-originPrev.pt.x, 2) + pow(kptPrev.pt.y-originPrev.pt.y, 2));
            const double distCurr = sqrt(pow(kptCurr.pt.x-originCurr.pt.x, 2) + pow(kptCurr.pt.y-originCurr.pt.y, 2));

            distRatios.push_back(distCurr / distPrev);
            
        }
    }
    

    // compute camera-based TTC from distance ratios
    double meanDistRatio = std::accumulate(distRatios.begin(), distRatios.end(), 0.0) / distRatios.size();
    double medianDistRatio;
    //array vector, and choose median value
    nth_element(distRatios.begin(), distRatios.begin() + distRatios.size() / 2, distRatios.end());
    medianDistRatio = distRatios[distRatios.size() / 2];
    double dT = 1 / frameRate;
    //TTC = -dT / (1 - meanDistRatio);
    TTC = -dT / (1-medianDistRatio);

    
}


void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
        // auxiliary variables
        double dT = 1/frameRate;
        double laneWidth = 4.0;

    // find closest distance to Lidar points within ego lane
    double minXPrev = 1e9, minXCurr = 1e9;
    
    //lidar point filtering by mean, standard deviation
    double sum_lidarPrev = 0;
    for (auto it = lidarPointsPrev.begin(); it != lidarPointsPrev.end(); ++it)
    {
        sum_lidarPrev += it->x; 
    }
    double mean_lidarPrev = sum_lidarPrev / lidarPointsPrev.size();
    //cout << "mean: "<< mean_lidarPrev << endl;
    double std_lidarPrev = 0;
    for (auto it = lidarPointsPrev.begin(); it != lidarPointsPrev.end(); ++it)
    {
        std_lidarPrev += (it->x - mean_lidarPrev) * (it->x - mean_lidarPrev) / lidarPointsPrev.size();
    }
    std_lidarPrev = sqrt (std_lidarPrev);
    //cout << "std lidarPrev" << std_lidarPrev << endl;

    double sum_lidarCurr = 0;
    for (auto it = lidarPointsCurr.begin(); it != lidarPointsCurr.end(); ++it)
    {
        sum_lidarCurr += it->x; 
    }
    double mean_lidarCurr = sum_lidarCurr / lidarPointsCurr.size();
    double std_lidarCurr = 0;
    for (auto it = lidarPointsCurr.begin(); it != lidarPointsCurr.end(); ++it)
    {
        std_lidarCurr += (it->x - mean_lidarCurr) * (it->x - mean_lidarCurr) / lidarPointsCurr.size();
    }
    std_lidarCurr = sqrt (std_lidarCurr);
    




    double std_multiple = 1.8;
    for (auto it = lidarPointsPrev.begin(); it != lidarPointsPrev.end(); ++it)
    {
        if(it->x >= mean_lidarPrev - (std_multiple * std_lidarPrev) && it->x <= mean_lidarPrev + (std_multiple * std_lidarPrev) )
        {
            minXPrev = minXPrev > it->x ? it->x : minXPrev;
        }
        
    }

    for (auto it = lidarPointsCurr.begin(); it != lidarPointsCurr.end(); ++it)
    {
        if(it->x >= mean_lidarCurr - (std_multiple * std_lidarCurr) && it->x <= mean_lidarCurr + (std_multiple * std_lidarCurr) )
        {
            minXCurr = minXCurr > it->x ? it->x : minXCurr;
        }
        
    }


    cout << "minXPrev: " << minXPrev << ", minXCurr: " << minXCurr << endl;
    // compute TTC from both measurements
    TTC = minXCurr * dT / (minXPrev - minXCurr);
}



void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{
    
    for (const auto prevBox : prevFrame.boundingBoxes)
    {
        int max_counter = 0;
        for (const auto currBox : currFrame.boundingBoxes)
        {
            int counter = 0;
            for (const auto match : matches)
            {
                const cv::Point2f prevPt = prevFrame.keypoints[match.queryIdx].pt;
                const cv::Point2f currPt = currFrame.keypoints[match.trainIdx].pt;

                if (prevBox.roi.contains(prevPt) && currBox.roi.contains(currPt))
                {
                    counter++;
                }
            }
            if (counter > max_counter)
            {
                max_counter = counter;
                bbBestMatches[prevBox.boxID] = currBox.boxID;
            }
        }
    }
}
