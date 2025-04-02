#include <iostream>
#include <depthai/depthai.hpp>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <vector>
#include <limits>
#include <numeric>
#include <map>
#include <set>
#include <queue>
#include <tuple>
#include <algorithm>
#include <string>
#include <sstream>
#include <chrono>
#include <array>

#include "misc.h"

/// Converting an array into Point2f
void arrayToPoint2f(std::vector< std::array<float, 3> >& gatheredPoints, std::vector<cv::Point2f>& gathered2fPoints){
	for( const std::array<float, 3>& point : gatheredPoints){
		gathered2fPoints.push_back( cv::Point2f(point[0], point[1]) );
	}
}


/// Temprorary way to find the corresponding Z values
float findMatchingValues(const cv::Point2f& pt, const std::vector<std::array<float, 3>>& arrayData, float tolerance) {
    for (const auto& arr : arrayData) {
        if (std::abs(pt.x - arr[0]) < tolerance && std::abs(pt.y - arr[1]) < tolerance) {
            return arr[2];
        }
    }
    return -1;
}



/// Graphing the points on the 2D image
void graphPoints( const std::vector<cv::Point2f>& hull, cv::Mat displayImage, double minDepth, bool final ){
    // Set up the display window and projection parameters
    int width = 1280, height = 720;
    cv::Mat image = displayImage.clone();

    // Scale the circle size based on minimal depth of the object to simulate depth
    int radius = static_cast<int>(3 / minDepth);
    radius = std::max(1, std::min(20, radius));    // Clamp radius between 1 and 20
    cv::Scalar color(255, 0, 0);
    
    // Draw each 3D point on the 2D image 
    std::vector<cv::Point> vertices;
    for (const cv::Point2f& pt2D : hull) {
        // Project the 3D point onto the 2D image plane
	    vertices.push_back(pt2D);

        // std::cout << "\t\tPoint: X=" << pt2D.x << " Y=" << pt2D.y << " Z=" << minDepth << std::endl; 
        cv::circle(image, pt2D, radius, color, -1);  // -1 fills the circle
    }

    // Display the result
    if( final == true ){
        cv::imshow("Final Hull", image);
        cv::waitKey(0);
    } else {
        cv::imshow("All points combined", image);
        // Draw contours
    	cv::polylines(image, vertices, true, color, 2);
    }

}

/// Projecting 3d points onto the 2d image plane
void projectPoints(const std::vector< std::array<float, 7> >& hull, std::vector<cv::Point2f>& projectedPoints) {
    // The structure: 0=X, 1=Y, 2=Z, 3=fx, 4=fy, 5=cx, 6=cy

    // The procedure
    for (const std::array<float, 7>& point : hull) {
        // TODO: try subsituting point[2] with minDepth to get better results
        float x = point[3] * (point[0] / point[2]) + point[5];
        float y = point[4] * (point[1] / point[2]) + point[6];

        projectedPoints.push_back(cv::Point2f(x, y));
    }

}

// Function for calculating median
double findMedian( std::vector<double> v, int n ){
    // Sort the vector
	std:sort(v.begin(), v.end());

    // Check if the number of elements is odd
    if (n % 2 != 0)
        return (double)v[n / 2];

    // If the number of elements is even, return the average
    // of the two middle elements
    return (double)(v[(n - 1) / 2] + v[n / 2]) / 2.0;
}


// Expands the bounding box of the front-facing points in X and Y by 'volume_increase_m'.
void expandFrontSides(std::vector< std::array<double, 3> >& points, float volume_increase_m, float newZ) {
    // Step 1: Identify bounding box in x,y for points in front of camera
    float minX =  std::numeric_limits<float>::max();
    float maxX = -std::numeric_limits<float>::max();
    float minY =  std::numeric_limits<float>::max();
    float maxY = -std::numeric_limits<float>::max();

    // Only consider points with z > 0
    for (const auto& p : points) {
        if (p[0] < minX) minX = p[0];
        if (p[0] > maxX) maxX = p[0];
        if (p[1] < minY) minY = p[1];
        if (p[1] > maxY) maxY = p[1];
    }

    // If failed to get a proper bounding box, do nothing
    if (minX > maxX || minY > maxY) {
        std::cout << "[ERROR] Failed to get a proper bounding box for expansion of the obstacle." << std::endl;
        return;
    }

    float widthX = maxX - minX;  // original X width
    float widthY = maxY - minY;  // original Y width
    if (widthX <= 0.0f || widthY <= 0.0f) {
        // Degenerate case: no real area to expand
        std::cout << "[ERROR] Area of the obstacle is 0. Failed to get a proper bounding box for expansion of the obstacle." << std::endl;
        return;
    }

    // Step 2: Compute the new (expanded) bounding box widths
    float newWidthX = widthX + volume_increase_m;
    float newWidthY = widthY + volume_increase_m;

    // Step 3: Compute how much to scale X and Y around the center
    float scaleX = newWidthX / widthX;
    float scaleY = newWidthY / widthY;

    float centerX = 0.5f * (minX + maxX);
    float centerY = 0.5f * (minY + maxY);

    // Step 4: Scale each front-facing point around (centerX, centerY).
    for (auto& p : points) {
        p[0] = centerX + (p[0] - centerX) * scaleX; // x
        p[1] = centerY + (p[1] - centerY) * scaleY; // y
        
	p[2] = p[2] > 0.35 ? p[2] - 0.35 : p[2]; // z
	p[2] += newZ;

	std::cout << "current: " << p[2] << "\n";
	std::cout << "newz: " << newZ << "\n\n";
    }
}
