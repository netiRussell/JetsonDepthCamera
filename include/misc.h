#pragma once

void arrayToPoint2f(std::vector< std::array<float, 3> >& gatheredPoints, std::vector<cv::Point2f>& gathered2fPoints);
float findMatchingValues(const cv::Point2f& pt, const std::vector<std::array<float, 3>>& arrayData, float tolerance = 1.0f);
double findMedian( std::vector<double> v, int n );
void expandFrontSides(std::vector< std::array<double, 3> >& points, float volume_increase_m, float newZ);
void projectPoints(const std::vector< std::array<float, 7> >& hull, std::vector<cv::Point2f>& projectedPoints);
void graphPoints( const std::vector<cv::Point2f>& hull, cv::Mat displayImage, double minDepth, bool final );
