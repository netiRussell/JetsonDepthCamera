// negative x => left; positive y => down.
// image is flipped in x-axis


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

#include "shortestPathAlg.h"
#include "misc.h"

std::map<Point, std::vector<Edge>> graph;

// Create a global GEOS context
GEOSContextHandle_t geos_ctx = nullptr;

std::vector< std::array<double, 3> > analyzeCaptures( std::vector< std::array<float, 3> >& gatheredPoints, double minDepth, cv::Mat displayImage, float fx, float fy, float cx, float cy, float newX, float newY, float newZ, float volume_increase_m);

std::vector< std::array<double, 3> > generateConvexHull(const int num_captures, const int minDepth, const int maxDepth, std::shared_ptr<dai::DataOutputQueue> depthQueue, float fx, float fy, float cx, float cy, float depthUnit, float newX, float newY, float newZ, float volume_increase_m);

std::array<double, 3> projectPointOntoPlane(const std::array<double, 3>& P,  const std::array<double, 3>& P0, const std::array<double, 3>& n);

struct HullData {
    std::vector<cv::Point> hull;
    double minDepth;
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

    stereo->setDepthAlign(dai::CameraBoardSocket::CAM_C);
    stereo->setOutputSize(1280, 720);
    

    // StereoDepth properties
    stereo->setDefaultProfilePreset(dai::node::StereoDepth::PresetMode::HIGH_DENSITY);
    stereo->initialConfig.setConfidenceThreshold(225);
    stereo->setLeftRightCheck(true);
    stereo->initialConfig.setMedianFilter(dai::MedianFilter::KERNEL_7x7);
    stereo->setExtendedDisparity(true);

    // Getting stereo depth units
    auto config = stereo -> initialConfig.get();
    auto depthUnitEnum = config.algorithmControl.depthUnit;
    float depthUnit = 1.f;
    switch(depthUnitEnum) {
        case dai::RawStereoDepthConfig::AlgorithmControl::DepthUnit::MILLIMETER:
            depthUnit = 0.001f;  // from mm to meters
            break;
        case dai::RawStereoDepthConfig::AlgorithmControl::DepthUnit::CENTIMETER:
            depthUnit = 0.01f;   // from cm to meters
            break;
        case dai::RawStereoDepthConfig::AlgorithmControl::DepthUnit::METER:
            depthUnit = 1.f;
            break;

        default:
            std::cout << "[WARNING] no corresponding depth unit has been found." << std::endl;
    }

	/*
    config.postProcessing.speckleFilter.enable = true; // If noise is nott too much - turn off
    config.postProcessing.speckleFilter.speckleRange = 60;
    config.postProcessing.temporalFilter.enable = true;
    config.postProcessing.spatialFilter.enable = true;
    config.postProcessing.spatialFilter.holeFillingRadius = 2;
    config.postProcessing.spatialFilter.numIterations = 1;
    config.postProcessing.decimationFilter.decimationFactor = 1; // turn off for the sake of perfomance
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
    /*
    auto distortionCoeffs = calibData.getDistortionCoefficients(dai::CameraBoardSocket::CAM_C);
    for (float value : distortionCoeffs){
    	std::cout << value << std::endl;
    }
    */

    // Extract intrinsic parameters
    float fx = intrinsics[0][0];
    float fy = intrinsics[1][1];
    float cx = intrinsics[0][2];
    float cy = intrinsics[1][2];

    // TODO: delete
    //std::cout << "cx = " << cx << " cy = " << cy << std::endl;

    // Depth thresholds in millimeters (1000mm = 1m)
    const int minDepth = 100;
    const int maxDepth = 750;

    // Source and end points
    static std::array<double, 3> source = {0, 0, 0};
    static std::array<double, 3> endp   = {0, 0, 1}; // go 100cm forward

    // Convex Hull & Shortest Path generation functionality
    const int num_captures = 20;
    const float volume_increase_m = 0.05;

    // -- First launch --
    // Record the start time
    auto start_time = std::chrono::high_resolution_clock::now();

    // Compute the convex hull
    std::vector< std::array<double, 3> > cube_vertices = generateConvexHull(num_captures, minDepth, maxDepth, depthQueue, fx, fy, cx, cy, depthUnit, 0, 0, 0, volume_increase_m);

    // Get the size of the first dimension (number of rows)
    int num_rows = cube_vertices.size();

    // Initialize GEOS
    geos_ctx = GEOS_init_r();

    Point vs[num_rows];
    for (int i=0;i<num_rows;i++) vs[i]=arrToPoint(cube_vertices[i]);
    Point s = arrToPoint(source);
    Point e = arrToPoint(endp);

    // Add convex hull vertices as nodes
    for (int i=0;i<num_rows;i++) add_node(vs[i], graph);
    add_node(s, graph);
    add_node(e, graph);

    // Add convex hull edges
    auto cube_edges = add_outer_edges_cube(num_rows, cube_vertices, graph);

    // Add edges from source
    //std::cout << "Adding edges from source:\n";
    add_edges_without_intersection(s, cube_edges, num_rows, cube_vertices, geos_ctx, graph);

    // Add edges from end
    //std::cout << "\nAdding edges from end:\n";
    add_edges_without_intersection(e, cube_edges, num_rows, cube_vertices, geos_ctx, graph);

    // A* search
    std::vector<Point> path = astar_path(s, e, graph);
    
    if (!path.empty()) {
        std::cout << "\nShortest path found by A* algorithm:\n[";
        for (auto &p: path) std::cout << fmtPoint(p) << " ";
        std::cout << "]\n";
    } else {
        std::cout << "No path found.\n";
    }

    GEOS_finish_r(geos_ctx);

    // Record the end time
    auto end_time = std::chrono::high_resolution_clock::now();
        
    // Calculate and print execution time
    std::chrono::duration<double> duration = end_time - start_time;
    std::cout << "\nExecution time: " << duration.count() << " seconds\n";

    // -- Secnd launch and beyound --
    int answr = 0;
    while( answr != 3 ){
        std::cout << "Press:\n\t1 to generate a new convex hull, \n\t2 to generate a new convex hull with a new position, \n\t3 if your want to quit the program.\n\nInput = ";
        std::cin >> answr;

        // Record the start time
        start_time = std::chrono::high_resolution_clock::now();

        // Empty the graph
        graph.clear();

        if( answr == 1 ){

            // Compute the convex hull
            cube_vertices = generateConvexHull(num_captures, minDepth, maxDepth, depthQueue, fx, fy, cx, cy, depthUnit, 0, 0, 0, volume_increase_m);
	    

        } else if( answr == 2 ){
            float newX, newY, newZ;
            std::cout << "The X of the new position = ";
            std::cin >> newX;
            std::cout << "The Y of the new position = ";
            std::cin >> newY;
            std::cout << "The Z of the new position = ";
            std::cin >> newZ;

	    // Update the source for the next position
            source = {newX, newY, newZ};

            // Compute the convex hull
            cube_vertices = generateConvexHull(num_captures, minDepth, maxDepth, depthQueue, fx, fy, cx, cy, depthUnit, newX, newY, newZ, volume_increase_m);
        } else{
            break;
        }
	
	// Get the size of the first dimension (number of rows)
    	num_rows = cube_vertices.size();

        // Initialize GEOS
        geos_ctx = GEOS_init_r();

        Point vs[num_rows];
        for (int i=0;i<num_rows;i++) vs[i]=arrToPoint(cube_vertices[i]);
        Point s = arrToPoint(source);
        Point e = arrToPoint(endp);

        // Add convex hull vertices as nodes
        for (int i=0;i<num_rows;i++) add_node(vs[i], graph);
        add_node(s, graph);
        add_node(e, graph);

        // Add convex hull edges
        cube_edges = add_outer_edges_cube(num_rows, cube_vertices, graph);

        // Add edges from source
        //std::cout << "Adding edges from source:\n";
        add_edges_without_intersection(s, cube_edges, num_rows, cube_vertices, geos_ctx, graph);

        // Add edges from end
        //std::cout << "\nAdding edges from end:\n";
        add_edges_without_intersection(e, cube_edges, num_rows, cube_vertices, geos_ctx, graph);

        // A* search
        path = astar_path(s, e, graph);
        
        if (!path.empty()) {
            std::cout << "\nShortest path found by A* algorithm:\n[";
            for (auto &p: path) std::cout << fmtPoint(p) << " ";
            std::cout << "]\n";
        } else {
            std::cout << "No path found.\n";
        }

        GEOS_finish_r(geos_ctx);

        // Record the end time
        end_time = std::chrono::high_resolution_clock::now();
            
        // Calculate and print execution time
        duration = end_time - start_time;
        std::cout << "\nExecution time: " << duration.count() << " seconds\n";
    }

    return 0;
}

std::vector< std::array<double, 3> > generateConvexHull(const int num_captures, const int minDepth, const int maxDepth, std::shared_ptr<dai::DataOutputQueue> depthQueue, float fx, float fy, float cx, float cy, float depthUnit, float newX, float newY, float newZ, float volume_increase_m){
    int counter = 1;
    int nCaptPerAnalysis = num_captures;
    std::vector<HullData> netHulls;
    std::vector<double> minDepths;
    std::vector< std::array<double, 3> > finalOutput;

    while (counter <= nCaptPerAnalysis) {
        // Get depth frame
        auto depthFrame = depthQueue->get<dai::ImgFrame>();
        cv::Mat depthImage = depthFrame->getFrame();
        
        // TODO: delete
        // Color and show the last capture
        cv::Mat depthImage8U;
        cv::normalize(depthImage, depthImage8U, 0, 255, cv::NORM_MINMAX, CV_8U);
        cv::applyColorMap(depthImage8U, depthImage8U, cv::COLORMAP_HOT);
        //cv::imshow("Last Capture", depthImage8U); 

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

	// Filter the mask
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
        cv::dilate(mask, mask, kernel);
        cv::erode(mask, mask, kernel);

	// If the raw depth is 0, we do not want to include that pixel:
	cv::Mat zeroDepthMask = (depthImage == 0);  // 8U mask of pixels that are 0 in depth
	mask.setTo(0, zeroDepthMask);   
     
        // Find contours in the mask
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(mask, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

        // Compute convex hulls for each contour
        std::vector<std::vector<cv::Point>> hulls;
	double minAreaThreshold = 500.0;
        for (size_t i = 0; i < contours.size(); i++) {
	    double area = cv::contourArea(contours[i]);
            if(contours[i].size() < 3 || area < minAreaThreshold) {
                // Not enough points for hull or it is too small; skip this contour.
		cv::drawContours(mask, std::vector<std::vector<cv::Point>>{contours[i]}, -1, cv::Scalar(0), cv::FILLED);
                continue;
            }
            std::vector<cv::Point> hull;
            cv::convexHull(contours[i], hull);
            hulls.push_back(hull);
        }

	// Find the minimal depth within the depthImage with the threshold mask
	double minDepthOfObj;	
        cv::minMaxLoc(depthImage, &minDepthOfObj, nullptr, nullptr, nullptr, mask);

        // Filter out noise
        for (size_t i = 0; i < hulls.size(); i++) {

            // Skip if the captured hull is just a noise
            if(hulls[i].empty()){
                continue;
            }

            // Initialize the hull for further analysis
            HullData newHullData;
            newHullData.hull = hulls[i];          // Assign the vector of points
            newHullData.minDepth = minDepthOfObj; // Set the minimal depth

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
	    // cv::imshow("Only the object", displayImage);


            // Draw convex hulls on the image
            for (size_t i = 0; i < hulls.size(); i++) {
            	cv::drawContours(displayImage, hulls, static_cast<int>(i), cv::Scalar(0, 255, 0), 2);
            }

	        // Display image
            //cv::imshow("Convex Hulls", displayImage);

            // ! TODO: change the structure to have all of the coordinates in a single capture
            // Filter out coordinates for visualization
            std::vector< std::array<float, 3> > points;

            for (int j = 0; j < netHulls.size(); j++) {

                minDepths.push_back(netHulls[j].minDepth);

                for (const auto& point : netHulls[j].hull) {
                    // 2d pixel coordinates
                    float x = point.x;
                    float y = point.y;

                    ushort depthValue = depthImage.at<ushort>(y, x);
                    if (depthValue == 0){
                        //std::cout << "[WARNING] depth value of x = " << x << ", y = " << y << " is invalid!" << std::endl;
                        continue; // ! TODO: make sure its ok to skip
                    }

                    float Z = static_cast<float>(depthValue)*depthUnit; // Convert mm to meters
                    // float X = (x - cx) * Z / fx;
                    // float Y = (y - cy) * Z / fy;

                    points.push_back({x, y, Z});
                }

            }

		std::cout << "Avg: " << std::reduce(minDepths.begin(), minDepths.end()) / minDepths.size() << std::endl;
            	
		double depth = 0;
		if(minDepths.size() != 0){
			depth = findMedian(minDepths, minDepths.size());
		} else {
			std::cout << "[ERROR] There are no convex hulls found";
			return {}; // Return an empty vector
		}

	        finalOutput = analyzeCaptures(points, depth, displayImage, fx, fy, cx, cy, newX, newY, newZ, volume_increase_m);
        }

        counter++;
    }

    cv::waitKey(100); // Wait briefly (100 ms) to allow resources to close
    cv::destroyAllWindows(); // Explicitly close all OpenCV windows

    return finalOutput;
}

/// Generating a final convex hull with as little vertices as possible from the gathered points
std::vector< std::array<double, 3> > analyzeCaptures( std::vector< std::array<float, 3> >& gatheredPoints, double minDepth, cv::Mat displayImage, float fx, float fy, float cx, float cy, float newX, float newY, float newZ, float volume_increase_m){

    minDepth /= 1000; // Convert mm to meters
    
    // Convert (x,y,z) points to (x,y)
    std::vector<cv::Point2f> gathered2fPoints;
    arrayToPoint2f(gatheredPoints, gathered2fPoints);

    // Graph all the points 
    //graphPoints(gathered2fPoints, displayImage, minDepth, false);

    // Compute the convex hull of the projected points and store vertices
    std::vector<cv::Point2f> finalHull;
    cv::convexHull(gathered2fPoints, finalHull);

    // Approximate the convex hull to get less vetices => better perfomance
    double arc_length = cv::arcLength(finalHull, true);
    std::vector<cv::Point2f> approxHull;
    double epsilon = 0.005 * arc_length; // can be played with to find the golden middle
    cv::approxPolyDP(finalHull, approxHull, epsilon, true);

    // Graph the final convex hull 
    graphPoints(approxHull, displayImage, minDepth, true);

    // -- Just the final points graph -----------------------------------
    // Set up the display window and projection parameters
    int width = 1280, height = 720;
    cv::Mat image = cv::Mat::zeros(height, width, CV_8UC3);

    // Scale the circle size based on minimal depth of the object to simulate depth
    int radius = static_cast<int>(3 / minDepth);
    radius = std::max(1, std::min(20, radius));    // Clamp radius between 1 and 20
    cv::Scalar color(255, 0, 0);

    // Draw each 3D point on the 2D image
    std::cout << "--------------------------------------------------------------------------\n\n\n\nApprox Coordinates:" << std::endl;
    std::vector< std::array<double, 3> > finalOutput;
    for (const cv::Point2f& pt2D : approxHull) {

        //TODO: exhaustive, to be changed:
        float Z = findMatchingValues(pt2D, gatheredPoints);
        if( Z == -1 ){
            std::cout << "[WARNING]: no Z corresponding found, skipping these x,y." << std::endl;
        }

        
        // Draw the point on the image
        cv::circle(image, pt2D, radius, color, -1);  // -1 fills the circle

        // Project the 2D points back to 3D
	    float x = (pt2D.x - cx) * Z / fx;
    	float y = (pt2D.y - cy) * Z / fy;

        // Text to display the coordinates
        cv::putText(image, "X= " + std::to_string(x) + "m, Y= " + std::to_string(-y) + "m, Z=" + std::to_string(Z), pt2D, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 1 );

        std::cout << "\tRelative Point: X=" << x  << "m, Y=" << -y  << "m, Z=" << Z << "m" << std::endl;

        // Add the points to the final output holder:
        x = x + newX;
        y = -(y) + newY;
        finalOutput.push_back({x, y, Z});

        std::cout << "\t\tUniversal Point: X=" << x << "m, Y=" << y << "m, Z=" << Z + newZ << "m" << std::endl;
        
    }

    std::cout << std::endl;

    // -- Bringing all of the points to the same Z level --
    // Find vertex with the smallest Z
    auto min_it = std::min_element(
        finalOutput.begin(), finalOutput.end(),
        [](const std::array<double, 3>& a, const std::array<double, 3>& b) {
            return a[2] < b[2];
        }
    );
    std::cout << "\nMin Z. X: " << (*min_it)[0] << "  Y: " << (*min_it)[1] << "  Z: " << (*min_it)[2] << std::endl;

    // Find normal with the camera position and the coordinate with the smallest Z
    // Camera position is (newX, newY, newZ)
    // point is (*min_it) = (x, y, Z)
    // Normal = (camera position - point with the smallest Z)
    std::array<double, 3> normal = {newX - (*min_it)[0], newY - (*min_it)[1], newZ - (*min_it)[2]};

    // Find intersection of the plane with every other point to create boundaries (vertices) of the final convex hull
    for (int i = 0; i < finalOutput.size(); i++)
    {
        std::cout << "\tUniversal Point: X=" << finalOutput[i][0] << "m, Y=" << finalOutput[i][1] << "m, Z=" << finalOutput[i][2] + newZ << "m" << std::endl;

        finalOutput[i] = projectPointOntoPlane( finalOutput[i], *min_it, normal );

        std::cout << "\Changed Point: X=" << finalOutput[i][0] << "m, Y=" << finalOutput[i][1] << "m, Z=" << finalOutput[i][2] + newZ << "m" << std::endl;
        std::cout << "\n";
    }

    expandFrontSides(finalOutput, volume_increase_m, newZ);

    // Display the result
    cv::imshow("Just the final points", image);
    cv::waitKey(0);

    return finalOutput;
}


// Project point P onto the plane defined by point P0 and normal n
std::array<double, 3> projectPointOntoPlane(const std::array<double, 3>& P,  const std::array<double, 3>& P0, const std::array<double, 3>& n) {
    // (P - P0) dot n
    std::array<double, 3> holder = {P[0] - P0[0], P[1] - P0[1], P[2] - P0[2]};
    double numerator = holder[0]*n[0] + holder[1]*n[1] + holder[2]*n[2];

    // n dot n
    double denominator = n[0]*n[0] + n[1]*n[1] + n[2]*n[2];

    // If denominator is zero, that means n = (0,0,0) ==> plane normal is invalid
    if (std::fabs(denominator) < 1e-9) {
        // Return P unchanged
        return P;
    }

    double t = -numerator / denominator;

    // Intersection point = P + (n * t)
    holder = {P[0] + (n[0] * t), P[1] + (n[1] * t), P[2] + (n[2] * t)};
    return holder;
}
