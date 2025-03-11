// negative x => left; positive y => down.
// image is flipped in x-axis
// TODO: make sure the path is updated
// TODO: make sure the camera is capable of capturing objects when they are rights in front of it
// TODO: lower filters for convex hulls

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
#include <geos_c.h>
#include <chrono>

ushort interpolateDepth(const cv::Mat& depthImage, int x, int y);
std::vector< std::array<double, 3> > analyzeCaptures( std::vector< std::array<float, 3> >& gatheredPoints, double minDepth, cv::Mat displayImage, float fx, float fy, float cx, float cy, float newX, float newY, float newZ);
void projectPoints(const std::vector< std::array<float, 7> >& hull, std::vector<cv::Point2f>& projectedPoints);
void graphPoints( const std::vector<cv::Point2f>& hull, cv::Mat displayImage, double minDepth, bool final );
std::vector< std::array<double, 3> > generateConvexHull(const int num_captures, const int minDepth, const int maxDepth, std::shared_ptr<dai::DataOutputQueue> depthQueue, float fx, float fy, float cx, float cy, float depthUnit, float newX, float newY, float newZ);
void arrayToPoint2f(std::vector< std::array<float, 3> >& gatheredPoints, std::vector<cv::Point2f>& gathered2fPoints);
float findMatchingValues(const cv::Point2f& pt, const std::vector<std::array<float, 3>>& arrayData, float tolerance = 1.0f);
double findMedian( std::vector<double> v, int n );

struct HullData {
    std::vector<cv::Point> hull;
    double minDepth;
};

// Shortest path finding component --------------------------------------------
struct Point {
    double x,y,z;
};

inline bool operator<(const Point &a, const Point &b) {
    if (a.x != b.x) return a.x < b.x;
    if (a.y != b.y) return a.y < b.y;
    return a.z < b.z;
}

inline bool operator==(const Point &a, const Point &b) {
    return (a.x == b.x && a.y == b.y && a.z == b.z);
}

struct Edge {
    Point node;
    double weight;
};

std::map<Point, std::vector<Edge>> graph;

// Create a global GEOS context
GEOSContextHandle_t geos_ctx = nullptr;

std::vector<Point> astar_path(const Point &start, const Point &goal);
void add_edges_without_intersection(const Point &point, const std::vector<std::pair<Point,Point>> &cube_edges, int num_rows, std::vector< std::array<double, 3> > cube_vertices);
std::vector<std::pair<Point,Point>> add_outer_edges_cube(int num_rows, std::vector< std::array<double, 3> > cube_vertices);
bool segments_intersect_no_touches_geos(const Point &A, const Point &B, const Point &C, const Point &D);
double distance3D(const Point &a, const Point &b);
double distance2D(const Point &a, const Point &b);
Point arrToPoint(const std::array<double, 3> arr);
void add_node(const Point &p);
void add_edge(const Point &u, const Point &v, double w);
std::string fmtPoint(const Point &p);
std::string fmtArray(const Point &p);
std::string fmtEdgeAsArrays(const Point &p1, const Point &p2);
// -------------------------------------------------------------------------


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
    const int maxDepth = 550;

    // Source and end points
    static std::array<double, 3> source = {0, 0, 0};
    static std::array<double, 3> endp   = {0, 0, 1}; // go 100cm forward

    // Convex Hull & Shortest Path generation functionality
    const int num_captures = 30;
    const int volume_increace_cm = 32;

    // -- First launch --
    // Record the start time
    auto start_time = std::chrono::high_resolution_clock::now();

    // Compute the convex hull
    std::vector< std::array<double, 3> > cube_vertices = generateConvexHull(num_captures, minDepth, maxDepth, depthQueue, fx, fy, cx, cy, depthUnit, 0, 0, 0);

    // Get the size of the first dimension (number of rows)
    int num_rows = sizeof(cube_vertices) / sizeof(cube_vertices[0]);

    // Initialize GEOS
    geos_ctx = GEOS_init_r();

    Point vs[num_rows];
    for (int i=0;i<num_rows;i++) vs[i]=arrToPoint(cube_vertices[i]);
    Point s = arrToPoint(source);
    Point e = arrToPoint(endp);

    // Add convex hull vertices as nodes
    for (int i=0;i<num_rows;i++) add_node(vs[i]);
    add_node(s);
    add_node(e);

    // Add convex hull edges
    auto cube_edges = add_outer_edges_cube(num_rows, cube_vertices);

    // Add edges from source
    //std::cout << "Adding edges from source:\n";
    add_edges_without_intersection(s, cube_edges, num_rows, cube_vertices);

    // Add edges from end
    //std::cout << "\nAdding edges from end:\n";
    add_edges_without_intersection(e, cube_edges, num_rows, cube_vertices);

    // A* search
    std::vector<Point> path = astar_path(s, e);
    
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
            cube_vertices = generateConvexHull(num_captures, minDepth, maxDepth, depthQueue, fx, fy, cx, cy, depthUnit, 0, 0, 0);
	    

        } else if( answr == 2 ){
            float newX, newY, newZ;
            std::cout << "The X of the new position = ";
            std::cin >> newX;
            std::cout << "The Y of the new position = ";
            std::cin >> newY;
            std::cout << "The Z of the new position = ";
            std::cin >> newZ;

            // Compute the convex hull
            cube_vertices = generateConvexHull(num_captures, minDepth, maxDepth, depthQueue, fx, fy, cx, cy, depthUnit, newX, newY, newZ);
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
        for (int i=0;i<num_rows;i++) add_node(vs[i]);
        add_node(s);
        add_node(e);

        // Add convex hull edges
        cube_edges = add_outer_edges_cube(num_rows, cube_vertices);

        // Add edges from source
        //std::cout << "Adding edges from source:\n";
        add_edges_without_intersection(s, cube_edges, num_rows, cube_vertices);

        // Add edges from end
        //std::cout << "\nAdding edges from end:\n";
        add_edges_without_intersection(e, cube_edges, num_rows, cube_vertices);

        // A* search
        path = astar_path(s, e);
        
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

std::vector< std::array<double, 3> > generateConvexHull(const int num_captures, const int minDepth, const int maxDepth, std::shared_ptr<dai::DataOutputQueue> depthQueue, float fx, float fy, float cx, float cy, float depthUnit, float newX, float newY, float newZ){
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
        cv::imshow("Last Capture", depthImage8U); 

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

        // Filter the mask
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
        cv::dilate(mask, mask, kernel);
        cv::erode(mask, mask, kernel);
        cv::GaussianBlur(mask, mask, cv::Size(5, 5), 0);

        // Find contours in the mask
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(mask, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);


        // Compute convex hulls for each contour
        std::vector<std::vector<cv::Point>> hulls;
        for (size_t i = 0; i < contours.size(); i++) {
            if(contours[i].size() < 3) {
                // Not enough points for hull; skip this contour.
                continue;
            }
            std::vector<cv::Point> hull;
            cv::convexHull(contours[i], hull);
            hulls.push_back(hull);
        }

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
	    cv::imshow("Only the object", displayImage);


            // Draw convex hulls on the image
            for (size_t i = 0; i < hulls.size(); i++) {
            	cv::drawContours(displayImage, hulls, static_cast<int>(i), cv::Scalar(0, 255, 0), 2);
            }

	        // Display image
            cv::imshow("Convex Hulls", displayImage);

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

                        // Try to interpolate depth from neighboring pixels
                        depthValue = interpolateDepth(depthImage, x, y);

                        if (depthValue == 0)
                            continue; // If still zero, skip the point
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
	        finalOutput = analyzeCaptures(points, depth, displayImage, fx, fy, cx, cy, newX, newY, newZ);
        }

        counter++;
    }

    cv::waitKey(100); // Wait briefly (100 ms) to allow resources to close
    cv::destroyAllWindows(); // Explicitly close all OpenCV windows

    return finalOutput;
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
std::vector< std::array<double, 3> > analyzeCaptures( std::vector< std::array<float, 3> >& gatheredPoints, double minDepth, cv::Mat displayImage, float fx, float fy, float cx, float cy, float newX, float newY, float newZ){

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
    double epsilon = 0.005 * arc_length; // can be played with to find the golden middle
    cv::approxPolyDP(finalHull, approxHull, epsilon, true);

    // Project the 2D points back to 3D
    std::vector<cv::Point2f> approxCoordinates(approxHull.size());
    for (int i = 0; i < approxHull.size(); i++){
	    //TODO: exhaustive, to be changed:
	    float Z = findMatchingValues(approxHull[i], gatheredPoints);
	    if( Z == -1 ){
		    std::cout << "[WARNING]: no Z corresponding found, skipping these x,y." << std::endl;
	    }

        approxCoordinates[i].x = (approxHull[i].x - cx) * minDepth / fx;
        approxCoordinates[i].y = (approxHull[i].y - cy) * minDepth / fy;
    }    

    // Graph the final convex hull 
    graphPoints(approxHull, displayImage, minDepth, true);


    // -- Additional graph but special case -----------------------------------
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
        
        // Draw the point on the image
        cv::circle(image, pt2D, radius, color, -1);  // -1 fills the circle
	    float x = (pt2D.x - cx) * minDepth / fx;
    	float y = (pt2D.y - cy) * minDepth / fy;
        cv::putText(image, "X= " + std::to_string(x) + "m, Y= " + std::to_string(-y) + "m, Z=" + std::to_string(minDepth), pt2D, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 1 );

        std::cout << "\tRelative Point: X=" << x  << "m, Y=" << -y  << "m, Z=" << minDepth << "m" << std::endl;

        // Add the points to the final output holder:
        x = x + newX;
        y = -(y + newY);
        float z = minDepth + newZ;
        finalOutput.push_back({x, y, z});

        std::cout << "\t\tUniversal Point: X=" << x + newX  << "m, Y=" << -(y + newY)  << "m, Z=" << minDepth + newZ << "m" << std::endl;
        
        /* Calculated from detphImage
        //TODO: exhaustive, to be changed:
	    float Z = findMatchingValues(pt2D, gatheredPoints);
	    if( Z == -1 ){
		    std::cout << "[WARNING]: no Z corresponding found, skipping these x,y." << std::endl;
	    }

        x = (pt2D.x - cx) * Z / fx;
    	y = (pt2D.y - cy) * Z / fy;
        cv::circle(image, cv::Point2f (x, y), radius, cv::Scalar (0, 0, 255), -1);  // -1 fills the circle
        cv::putText(image, "X= " + std::to_string(x) + "m, Y= " + std::to_string(y) + "m, Z=" + std::to_string(Z), cv::Point2f (x, y), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 1 );

        std::cout << "\t\tReal Point: X=" << x << "m, Y=" << y << "m, Z=" << Z << "m" << "\n" << std::endl;
        */
    }
    

    // Display the result
    cv::imshow("Just the final points", image);
    cv::waitKey(0);

    std::cout << std::endl;

    return finalOutput;
}


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



// Shortest path finding components --------------------------------------------
Point arrToPoint(const std::array<double, 3> arr) {
    Point p; 
    p.x=arr[0]; 
    p.y=arr[1]; 
    p.z=arr[2]; 
    return p;
}

void add_node(const Point &p) {
    if (graph.find(p)==graph.end()) graph[p]=std::vector<Edge>();
}

void add_edge(const Point &u, const Point &v, double w) {
    graph[u].push_back({v,w});
    graph[v].push_back({u,w});
}


std::string fmtPoint(const Point &p) {
    std::ostringstream oss;
    oss << "(" << p.x << ", " << p.y << ", " << p.z << ")";
    return oss.str();
}


std::string fmtArray(const Point &p) {
    std::ostringstream oss;
    oss << "array([" << p.x << ", " << p.y << ", " << p.z << "])";
    return oss.str();
}

std::string fmtEdgeAsArrays(const Point &p1, const Point &p2) {
    std::ostringstream oss;
    oss << "(" << fmtArray(p1) << ", " << fmtArray(p2) << ")";
    return oss.str();
}

// Distance in 2D for weights
double distance2D(const Point &a, const Point &b) {
    double dx=a.x-b.x; double dy=a.y-b.y;
    return std::sqrt(dx*dx+dy*dy);
}

// Distance in 3D for heuristic
double distance3D(const Point &a, const Point &b) {
    double dx=a.x-b.x; double dy=a.y-b.y; double dz=a.z-b.z;
    return std::sqrt(dx*dx+dy*dy+dz*dz);
}

// Use GEOS to check intersection
bool segments_intersect_no_touches_geos(const Point &A, const Point &B, const Point &C, const Point &D) {
    // Create line segment for AB
    GEOSCoordSequence* seq1 = GEOSCoordSeq_create_r(geos_ctx, 2, 2);
    GEOSCoordSeq_setXY_r(geos_ctx, seq1, 0, A.x, A.y);
    GEOSCoordSeq_setXY_r(geos_ctx, seq1, 1, B.x, B.y);
    GEOSGeometry* line1 = GEOSGeom_createLineString_r(geos_ctx, seq1);

    // Create line segment for CD
    GEOSCoordSequence* seq2 = GEOSCoordSeq_create_r(geos_ctx, 2, 2);
    GEOSCoordSeq_setXY_r(geos_ctx, seq2, 0, C.x, C.y);
    GEOSCoordSeq_setXY_r(geos_ctx, seq2, 1, D.x, D.y);
    GEOSGeometry* line2 = GEOSGeom_createLineString_r(geos_ctx, seq2);

    char intersects = GEOSIntersects_r(geos_ctx, line1, line2);
    char touches = GEOSTouches_r(geos_ctx, line1, line2);

    bool result = false;
    if (intersects && !touches) {
        result = true;
    }

    GEOSGeom_destroy_r(geos_ctx, line1);
    GEOSGeom_destroy_r(geos_ctx, line2);

    return result;
}

// Connect consecutive convex hull points.
std::vector<std::pair<Point,Point>> add_outer_edges_cube(int num_rows, std::vector< std::array<double, 3> > cube_vertices) {
    Point vs[num_rows];
    for (int i=0;i<num_rows;i++) vs[i]=arrToPoint(cube_vertices[i]);
    std::vector<std::pair<Point,Point>> edges;
    for (int i=0; i<num_rows; i++) {
        int j = (i + 1) % num_rows;
        edges.push_back({vs[i], vs[j]});
        double w = distance2D(vs[i], vs[j]);
        add_edge(vs[i], vs[j], w);
    }
    return edges;
}

void add_edges_without_intersection(const Point &point, const std::vector<std::pair<Point,Point>> &cube_edges, int num_rows, std::vector< std::array<double, 3> > cube_vertices) {
    Point vs[num_rows];
    for (int i=0;i<num_rows;i++) vs[i]=arrToPoint(cube_vertices[i]);

    for (int i=0;i<num_rows;i++) {
        Point vertex = vs[i];
        bool intersects = false;
        for (auto &ce: cube_edges) {
            if (segments_intersect_no_touches_geos(point, vertex, ce.first, ce.second)) {
                //std::cout << "Edge from " << fmtPoint(point) << " to " << fmtPoint(vertex)
                //          << " intersects with cube edge " << fmtEdgeAsArrays(ce.first, ce.second) << "\n";
                intersects = true;
                break;
            }
        }
        if (!intersects) {
            double w = distance2D(point, vertex);
            add_edge(point, vertex, w);
            //std::cout << "Added edge from " << fmtPoint(point) << " to " << fmtPoint(vertex) << "\n";
        }
    }
}

std::vector<Point> astar_path(const Point &start, const Point &goal) {
    std::map<Point,double> gScore;
    std::map<Point,double> fScore;
    std::map<Point,Point> cameFrom;

    for (auto &kv: graph) {
        gScore[kv.first] = std::numeric_limits<double>::infinity();
        fScore[kv.first] = std::numeric_limits<double>::infinity();
    }

    gScore[start] = 0.0;
    fScore[start] = distance3D(start, goal);

   
    static long long counter = 0;

    struct PQItem {
        Point node;
        double f;
        long long order;
        bool operator>(const PQItem &o) const {
            if (f == o.f) return order > o.order;
            return f > o.f;
        }
    };

    std::priority_queue<PQItem, std::vector<PQItem>, std::greater<PQItem>> openSet;
    openSet.push({start, fScore[start], counter++});
    std::map<Point, bool> inOpen;
    inOpen[start] = true;

    while (!openSet.empty()) {
        Point current = openSet.top().node;
        openSet.pop();
        inOpen[current] = false;

        if (current == goal) {
            std::vector<Point> path;
            Point cur = current;
            while (!(cur == start)) {
                path.push_back(cur);
                cur = cameFrom[cur];
            }
            path.push_back(start);
            std::reverse(path.begin(), path.end());
            return path;
        }

        for (auto &edge: graph[current]) {
            Point neighbor = edge.node;
            double tentative = gScore[current] + edge.weight;
            if (tentative < gScore[neighbor]) {
                cameFrom[neighbor] = current;
                gScore[neighbor] = tentative;
                fScore[neighbor] = tentative + distance3D(neighbor, goal);
                if (!inOpen[neighbor]) {
                    openSet.push({neighbor, fScore[neighbor], counter++});
                    inOpen[neighbor] = true;
                }
            }
        }
    }

    return {};
}
// -------------------------------------------------------------------------

