﻿// TODO: Calibrate the camera to get the rectified intrinsic parameters
// TODO: Make sure the intrinsics are rectified by checking the distortion coefficients and the intrinsics values offset from the center
// TODO: Check if the perfomance is higher with the original or the minimal depth values (Zs)


#include <iostream>
#include <depthai/depthai.hpp>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <vector>
#include <limits>
#include <numeric>

ushort interpolateDepth(const cv::Mat& depthImage, int x, int y);
void analyzeCaptures( std::vector< std::array<float, 3> >& gatheredPoints, double minDepth, cv::Mat displayImage, float fx, float fy, float cx, float cy);
void projectPoints(const std::vector< std::array<float, 7> >& hull, std::vector<cv::Point2f>& projectedPoints);
void graphPoints( const std::vector<cv::Point2f>& hull, cv::Mat displayImage, double minDepth, bool final );
void generateConvexHull(const int num_captures, const int minDepth, const int maxDepth, std::shared_ptr<dai::DataOutputQueue> depthQueue, float fx, float fy, float cx, float cy);
void arrayToPoint2f(std::vector< std::array<float, 3> >& gatheredPoints, std::vector<cv::Point2f>& gathered2fPoints);
float findMatchingValues( const cv::Point2f& points, const std::vector<std::array<float, 3>>& arrayData );


struct HullData {
    std::vector<cv::Point> hull;
    bool isRealObject;
    double area;
    std::pair<double, double> coordinates;
};


int main() {
    // Create pipeline
    dai::Pipeline pipeline;

    // Define sources and outputs
    auto monoLeft = pipeline.create<dai::node::MonoCamera>();
    auto monoRight = pipeline.create<dai::node::MonoCamera>();

    auto stereo = pipeline.create<dai::node::StereoDepth>();
    auto xoutDepth = pipeline.create<dai::node::XLinkOut>();

    xoutDepth->setStreamName("depth");

    // Properties
    monoLeft->setResolution(dai::MonoCameraProperties::SensorResolution::THE_720_P);
    monoLeft->setBoardSocket(dai::CameraBoardSocket::CAM_B);

    monoRight->setResolution(dai::MonoCameraProperties::SensorResolution::THE_720_P);
    monoRight->setBoardSocket(dai::CameraBoardSocket::CAM_C);

    // StereoDepth properties
    stereo->setDefaultProfilePreset(dai::node::StereoDepth::PresetMode::HIGH_DENSITY);
    stereo->initialConfig.setConfidenceThreshold(225);
    stereo->setLeftRightCheck(true);
    stereo->initialConfig.setMedianFilter(dai::MedianFilter::KERNEL_7x7);
    stereo->setExtendedDisparity(true);

    /* Filters - can be used to improve the final output
    auto config = stereo->initialConfig.get();
    config.postProcessing.speckleFilter.enable = false;
    config.postProcessing.speckleFilter.speckleRange = 60;
    config.postProcessing.temporalFilter.enable = true;
    config.postProcessing.spatialFilter.enable = true;
    config.postProcessing.spatialFilter.holeFillingRadius = 2;
    config.postProcessing.spatialFilter.numIterations = 1;
    config.postProcessing.decimationFilter.decimationFactor = 1; // TODO: delete for the sake of perfomance
    stereo->initialConfig.set(config);
    */

    // Linking
    monoLeft->out.link(stereo->left);
    monoRight->out.link(stereo->right);
    stereo->depth.link(xoutDepth->input);

    // Connect to device and start pipeline
    dai::Device device(pipeline);

    // Get output queue
    auto depthQueue = device.getOutputQueue("depth", 4, false);

    // Retrieve calibration data
    auto calibData = device.readCalibration();
    auto intrinsics = calibData.getCameraIntrinsics(dai::CameraBoardSocket::CAM_C, 1280, 720);
    
    // TODO: delete
    auto distortionCoeffs = calibData.getDistortionCoefficients(dai::CameraBoardSocket::CAM_C);
    for (float value : distortionCoeffs){
    	std::cout << value << std::endl;
    }

    // Extract intrinsic parameters
    float fx = intrinsics[0][0];
    float fy = intrinsics[1][1];
    float cx = intrinsics[0][2];
    float cy = intrinsics[1][2];

    // TODO: delete
    std::cout << "cx = " << cx << " cy = " << cy << std::endl;

    // Depth thresholds in millimeters
    const int minDepth = 100;  // 0.1 meters
    const int maxDepth = 450; // 0.4 meters

    // Convex Hull & Shortest Path generation functionality
    const int num_captures = 40;
    generateConvexHull(num_captures, minDepth, maxDepth, depthQueue, fx, fy, cx, cy);

    int answr = 0;
    while( answr != 2 ){
        std::cout << "Press 1 to generate a new convex hull, 2 to quit the program: ";
        std::cin >> answr;

        if( answr == 1 ){
            generateConvexHull(num_captures, minDepth, maxDepth, depthQueue, fx, fy, cx, cy);
        } else if( answr == 2 ){
            break;
        }
    }

    return 0;
}

void generateConvexHull(const int num_captures, const int minDepth, const int maxDepth, std::shared_ptr<dai::DataOutputQueue> depthQueue, float fx, float fy, float cx, float cy){
    int counter = 1;
    int nCaptPerAnalysis = num_captures;
    std::vector<HullData> netHulls;
    std::vector<double> minDepths;

    while (counter <= nCaptPerAnalysis) {
        // Get depth frame
        auto depthFrame = depthQueue->get<dai::ImgFrame>();
        cv::Mat depthImage = depthFrame->getFrame();

        // Threshold depth image to create mask
        cv::Mat mask;
        cv::inRange(depthImage, minDepth, maxDepth, mask);

        // Check if the mask is empty (i.e., all pixels are 0)
	if(cv::countNonZero(mask) == 0) {
    		std::cout << "[WARNING] No pixels are found in the [" 
              	<< minDepth << ", " << maxDepth 
              	<< "] range. Skipping this iteration.\n";
   	 continue; // Skip this iteration, since there's no object in range
	}

        // Find the minimal depth within the depthImage with the threshold mask
        double minDepthOfObj;
        cv::minMaxLoc(depthImage, &minDepthOfObj, nullptr, nullptr, nullptr, mask);
        minDepths.push_back(minDepthOfObj);

        // Find contours in the mask
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(mask, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

	// Check if no contours are found
	if (contours.empty()) {
   		std::cout << "[WARNING] No valid contours are found. Skipping this iteration.\n";
    		continue;
	}

        // Compute convex hulls for each contour
        std::vector<std::vector<cv::Point>> hulls(contours.size());
        for (size_t i = 0; i < contours.size(); i++) {
            cv::convexHull(contours[i], hulls[i]);
        }

        // Filter out noise
        for (size_t i = 0; i < hulls.size(); i++) {

            std::vector< std::array<float, 7> > points; // Points of a single convex hull

            // Compute area in pixels and skip if the captured hull is just a noise
            double area = cv::contourArea(hulls[i]);
            if( area < 500 ){
                continue;
            }


            // Initialize the hull for further analysis
            HullData newHullData;
            newHullData.hull = hulls[i];          // Assign the vector of points
            newHullData.isRealObject = false;     // Set the flag
            newHullData.area = area;        	  // Set the area

            // Calculate the the center of the current hull
            cv::Moments mu = cv::moments(hulls[i]);
            newHullData.coordinates = std::make_pair(mu.m10 / mu.m00, mu.m01 / mu.m00); // x, y

            netHulls.push_back(newHullData);

        }

        // Analyze and clear the hulls that don't represent real objects
        if( counter == nCaptPerAnalysis ){
            // Create an image to display
            cv::Mat displayImage;
            cv::normalize(mask, displayImage, 0, 255, cv::NORM_MINMAX, CV_8UC1);
            cv::equalizeHist(displayImage, displayImage);
            cv::applyColorMap(displayImage, displayImage, cv::COLORMAP_HOT);

            // TODO: change bg color based on minDepth
            // displayImage.setTo( cv::Scalar(0, 0, 139) );
	        cv::imshow("Just the object", displayImage);


            // Draw convex hulls on the image
            for (size_t i = 0; i < hulls.size(); i++) {
            	cv::drawContours(displayImage, hulls, static_cast<int>(i), cv::Scalar(0, 255, 0), 2);
            }

	        // Display image
            cv::imshow("Convex Hulls", displayImage);

            // Check if the hull represents a real object
            for (size_t j = 0; j < netHulls.size(); j++) {

                // Check on other hulls
                // std::cout << "Current Area = " << netHulls[j].area << std::endl;
                for (size_t k = j+1; k < netHulls.size(); k++) {

                    // Make sure the hull hasn't been recognized yet
                    if( netHulls[k].isRealObject == true ){
                            continue;
                    }

                    // Make sure the area and the coordinates are similar
                    // ! TODO: play with the number to increase precision
                    if( fabs(netHulls[j].area - netHulls[k].area) <= 100 && (fabs(netHulls[j].coordinates.first - netHulls[k].coordinates.first) <= 0.5 && fabs(netHulls[j].coordinates.second - netHulls[k].coordinates.second) <= 0.5) ){
                        netHulls[j].isRealObject = true;
                        netHulls[k].isRealObject = true;

                        // std::cout << "\t Considered Area = " << netHulls[k].area << std::endl;
                        // std::cout << "Hull " << j << ": Centroid = (" << netHulls[j].coordinates.first << ", " << netHulls[j].coordinates.second << ")" << std::endl;
                    }

                }

            }


            // ! TODO: change the structure to have all of the coordinates in a single capture
            // Filter out coordinates for visualization
            std::vector< std::array<float, 3> > points;

            for (int j = 0; j < netHulls.size(); j++) {

                // Make sure the hull is recognized
                if( netHulls[j].isRealObject == false ){
                        continue;
                }

                for (const auto& point : netHulls[j].hull) {
                    // 2d pixel coordinates
                    float x = point.x;
                    float y = point.y;

                    ushort depthValue = depthImage.at<ushort>(y, x);
                    if (depthValue == 0){
                        // Try to interpolate depth from neighboring pixels
                        depthValue = interpolateDepth(depthImage, x, y);

                        if (depthValue == 0)
                            continue; // If still zero, skip the point
                    }

                    float Z = static_cast<float>(depthValue) / 1000.0f; // Convert mm to meters
                    // float X = (x - cx) * Z / fx;
                    // float Y = (y - cy) * Z / fy;

                    points.push_back({x, y, Z});
                }

            }

            analyzeCaptures(points, std::reduce(minDepths.begin(), minDepths.end()) / minDepths.size(), displayImage, fx, fy, cx, cy);
        }

        counter++;
    }

    cv::waitKey(100); // Wait briefly (100 ms) to allow resources to close
    cv::destroyAllWindows(); // Explicitly close all OpenCV windows
}

// TODO: redundant? Delete if yes
ushort interpolateDepth(const cv::Mat& depthImage, int x, int y) {
    int neighborhoodSize = 5;
    int count = 0;
    int sum = 0;

    for (int dy = -neighborhoodSize; dy <= neighborhoodSize; ++dy) {
        for (int dx = -neighborhoodSize; dx <= neighborhoodSize; ++dx) {
            int nx = x + dx;
            int ny = y + dy;
            // Check bounds
            if (nx >= 0 && nx < depthImage.cols && ny >= 0 && ny < depthImage.rows) {
                ushort neighborDepth = depthImage.at<ushort>(ny, nx);
                if (neighborDepth != 0) {
                    sum += neighborDepth;
                    ++count;
                }
            }
        }
    }

    if (count > 0)
        return static_cast<ushort>(sum / count);
    else
        return 0; // Unable to interpolate
}

/// Generating a final convex hull with as little vertices as possible from the gathered points
void analyzeCaptures( std::vector< std::array<float, 3> >& gatheredPoints, double minDepth, cv::Mat displayImage, float fx, float fy, float cx, float cy){

    minDepth /= 1000; // Convert mm to meters
    
    // Convert (x,y,z) points to (x,y)
    std::vector<cv::Point2f> gathered2fPoints;
    arrayToPoint2f(gatheredPoints, gathered2fPoints);

    // Graph all the points 
    graphPoints(gathered2fPoints, displayImage, minDepth, false);

    // Compute the convex hull of the projected points and store vertices
    std::vector<cv::Point2f> finalHull;
    cv::convexHull(gathered2fPoints, finalHull);

    // Approximate the convex hull to get less vetices => better perfomance
    double arc_length = cv::arcLength(finalHull, true);
    std::vector<cv::Point2f> approxHull;
    cv::approxPolyDP(finalHull, approxHull, 0.01*arc_length, true); // epsilon 1% to 5% of the arc length

    // Project the 2D points back to 3D
    std::vector<cv::Point2f> approxCoordinates(approxHull.size());
    for (int i = 0; i < approxHull.size(); i++){
	    //TODO: exhaustive, to be changed:
	    float Z = findMatchingValues(approxHull[i], gatheredPoints);
	    if( Z == -1 ){
		    std::cout << "[WARNING]: no Z corresponding found, skipping these x,y." << std::endl;
	    }

    	    approxCoordinates[i].x = (approxHull[i].x - cx) * Z / fx;
    	    approxCoordinates[i].y = (approxHull[i].y - cy) * Z / fy;
    }    

    std::cout << "--------------------------------------------------------------------------\n\n\n\nApprox Coordinates:" << std::endl;
    for( const cv::Point2f &coordinates : approxCoordinates ){
            std::cout << "\t\tPoint: X=" << coordinates.x << "m, Y=" << coordinates.y << "m, Z=" << minDepth << "m" << std::endl;
    }

    // Graph the final convex hull 
    graphPoints(approxHull, displayImage, minDepth, true);


    // -- Additional graph but special case -----------------------------------
    // Set up the display window and projection parameters
    int width = 1280, height = 720;
    cv::Mat justPoints = cv::Mat::zeros(height, width, CV_8UC3);

    // Scale the circle size based on minimal depth of the object to simulate depth
    int radius = static_cast<int>(3 / minDepth);
    radius = std::max(1, std::min(20, radius));    // Clamp radius between 1 and 20
    cv::Scalar color(255, 0, 0);

    // Draw each 3D point on the 2D image 
    std::vector<cv::Point> vertices;
    for (const cv::Point2f& pt2D : approxHull) {
        // Project the 3D point onto the 2D image plane
	    vertices.push_back(pt2D);

        // std::cout << "\t\tPoint: X=" << pt2D.x << " Y=" << pt2D.y << " Z=" << minDepth << std::endl; 
        cv::circle(justPoints, pt2D, radius, color, -1);  // -1 fills the circle
    }

    // Display the result
    cv::imshow("Just the final points", justPoints);
    cv::waitKey(0);


    std::cout << std::endl;
}


/// Converting an array into Point2f
void arrayToPoint2f(std::vector< std::array<float, 3> >& gatheredPoints, std::vector<cv::Point2f>& gathered2fPoints){
	for( const std::array<float, 3>& point : gatheredPoints){
		gathered2fPoints.push_back( cv::Point2f(point[0], point[1]) );
	}
}


/// Temprorary way to find the corresponding Z values
float findMatchingValues( const cv::Point2f& points, const std::vector<std::array<float, 3>>& arrayData ) {
    // For each point, search exhaustively in arrayData
    for (const auto& arr : arrayData) {
        // Compare x and y
        if (points.x == arr[0] && points.y == arr[1]) {
            // If there's a match, return the value
            return arr[2];
    }
	}

    // No match case
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

