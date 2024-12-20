#include <iostream>
#include <depthai/depthai.hpp>
#include <opencv2/opencv.hpp>
#include <cmath>

ushort interpolateDepth(const cv::Mat& depthImage, int x, int y);
void analyzeCaptures( std::vector< std::array<float, 7> >& gatheredPoints, double minDepth, cv::Mat displayImage );
void projectPoints(const std::vector< std::array<float, 7> >& hull, std::vector<cv::Point2f>& projectedPoints);
void graphPoints( const std::vector<cv::Point2f>& hull, cv::Mat displayImage, double minDepth, bool final );

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
    stereo->setDefaultProfilePreset(dai::node::StereoDepth::PresetMode::DEFAULT);
    stereo->initialConfig.setConfidenceThreshold(225);
    stereo->setLeftRightCheck(true);
    stereo->initialConfig.setMedianFilter(dai::MedianFilter::KERNEL_7x7);
    stereo->setExtendedDisparity(true);
    stereo->setSubpixel(false);

    auto config = stereo->initialConfig.get();

    config.postProcessing.speckleFilter.enable = false;

    config.postProcessing.speckleFilter.speckleRange = 50;

    config.postProcessing.temporalFilter.enable = true;

    config.postProcessing.spatialFilter.enable = true;

    config.postProcessing.spatialFilter.holeFillingRadius = 2;

    config.postProcessing.spatialFilter.numIterations = 1;

    config.postProcessing.decimationFilter.decimationFactor = 1;
    
    config.postProcessing.thresholdFilter.minRange = 100;

    config.postProcessing.thresholdFilter.maxRange = 700;

    stereo->initialConfig.set(config);

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

    // Extract intrinsic parameters
    float fx = intrinsics[0][0];
    float fy = intrinsics[1][1];
    float cx = intrinsics[0][2];
    float cy = intrinsics[1][2];

    // Depth thresholds in millimeters
    int minDepth = 100;  // 0.1 meters
    int maxDepth = 700; // 0.7 meters

    struct HullData {
        std::vector<cv::Point> hull;
        bool isRealObject;
        double area;
        std::pair<double, double> coordinates;
    };


    int i = 1;
    int nCaptPerAnalysis = 40;
    std::vector<HullData> netHulls;
    while (i <= nCaptPerAnalysis) {
        // Get depth frame
        auto depthFrame = depthQueue->get<dai::ImgFrame>();
        cv::Mat depthImage = depthFrame->getFrame();

        // Threshold depth image to create mask
        cv::Mat mask;
        cv::inRange(depthImage, minDepth, maxDepth, mask);

        // Find the minimal depth within the mask
        double minDepthInMask;
        cv::minMaxLoc(depthImage, &minDepthInMask, nullptr, nullptr, nullptr, mask);

        // Noise reduction
        //cv::erode(mask, mask, cv::Mat(), cv::Point(-1, -1), 2);
        //cv::dilate(mask, mask, cv::Mat(), cv::Point(-1, -1), 2);

        // Blur
        //cv::GaussianBlur(mask, mask, cv::Size(9, 9), 5);


        // Find contours in the mask
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(mask, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

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

            // Calculate the coordinates
            cv::Moments mu = cv::moments(hulls[i]);
            newHullData.coordinates = std::make_pair(mu.m10 / mu.m00, mu.m01 / mu.m00); // x, y

            netHulls.push_back(newHullData);

        }

        // Analyze and clear the hulls that don't represent real objects
        if( i == nCaptPerAnalysis-1 ){
            // Create an image to display
            cv::Mat displayImage;
            cv::normalize(depthImage, displayImage, 0, 255, cv::NORM_MINMAX, CV_8UC1);
            cv::equalizeHist(displayImage, displayImage);
            cv::applyColorMap(displayImage, displayImage, cv::COLORMAP_HOT);

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
            std::vector< std::array<float, 7> > points;

            for (int j = 0; j < netHulls.size(); j++) {

                // Make sure the hull is recognized
                if( netHulls[j].isRealObject == false ){
                        continue;
                }

                for (const auto& point : netHulls[j].hull) {
                    int x = point.x;
                    int y = point.y;

                    ushort depthValue = depthImage.at<ushort>(y, x);
                    if (depthValue == 0){
                        // Try to interpolate depth from neighboring pixels
                        depthValue = interpolateDepth(depthImage, x, y);

                        if (depthValue == 0)
                            continue; // If still zero, skip the point
                    }

                    float Z = static_cast<float>(depthValue) / 1000.0f; // Convert mm to meters
                    float X = (x - cx) * Z / fx;
                    float Y = (y - cy) * Z / fy;

                    points.push_back({X, Y, Z, fx, fy, cx, cy});
                }

            }

            analyzeCaptures(points, minDepthInMask, displayImage);

            // To restart the loop:
            // i = 1;
            // Clear the netHulls vector: netHulls.clear();
        }

        i++;
    }

    netHulls.clear();
    cv::waitKey(100); // Wait briefly (100 ms) to allow resources to close
    cv::destroyAllWindows(); // Explicitly close all OpenCV windows

    return 0;
}


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


void analyzeCaptures( std::vector< std::array<float, 7> >& gatheredPoints, double minDepth, cv::Mat displayImage){
    // Shape of gatheredCaptures = # of captures, # of convex hulls, # of coordinates, 5 coordinatex - X, Y, Z, cx, cy.
    std::cout << "\n------------------------------------------------------------------------\n";

    // Main loop of the function
    std::cout << "\tCurrent Convex Hull size = " << gatheredPoints.size() << "\n";

    // Print out the minimal depth
    std::cout << "--------------------------------------------------------------------------\n\n\n\nThe minimal depth of the object: " << minDepth << "mm" << std::endl;
    minDepth /= 1000;

    // Project the resulted points
    std::vector<cv::Point2f> projectedPoints;
    projectPoints(gatheredPoints, projectedPoints);
    
    // Graph all the points 
    graphPoints(projectedPoints, displayImage, minDepth, false);

    // Compute the final convex hull for visualizing (optional)
    std::vector<cv::Point2f> finalHull;
    cv::convexHull(projectedPoints, finalHull);

    double arc_length = cv::arcLength(finalHull, true);
    std::vector<cv::Point2f> approxHull;
    cv::approxPolyDP(finalHull, approxHull, 0.01*arc_length, true); // epsilon 1% to 5% of the arc length
    
    // Compute the final convex hull for coordinates
    std::vector<int> hullIndices;
    cv::convexHull(projectedPoints, hullIndices, false, false);

    // Gather the coordinates found
    std::vector<std::array<float, 7>> finalCoordinates;
    for (int idx : hullIndices) {
        gatheredPoints[idx][2] = minDepth;
        finalCoordinates.push_back(gatheredPoints[idx]); // Includes X, Y, Z, fx, fy, cx, cy
    }

    // Getting the original version of the approximated coordinates 
    // TODO: (Super-duper stupid, must be changed)
    std::vector<std::array<float, 7>> approxCoordinates;
    for (const auto& point : approxHull) {
        for (size_t i = 0; i < finalHull.size(); ++i) {
            if (cv::norm(point - finalHull[i]) < 1e-6) { // Find closest match
                approxCoordinates.push_back(finalCoordinates[i]); // Get index in gatheredPoints
                break;
            }
        }
    }

    // Print the final coordinates
    std::cout << "--------------------------------------------------------------------------\n\n\n\nFinal Coordinates:" << std::endl;
    for( const std::array<float, 7> &coordinates : finalCoordinates ){
            std::cout << "\t\tPoint: X=" << coordinates[0] << "m, Y=" << coordinates[1] << "m, Z=" << coordinates[2] << "m" << std::endl;
    }

    std::cout << "--------------------------------------------------------------------------\n\n\n\nApprox Coordinates:" << std::endl;
    for( const std::array<float, 7> &coordinates : approxCoordinates ){
            std::cout << "\t\tPoint: X=" << coordinates[0] << "m, Y=" << coordinates[1] << "m, Z=" << coordinates[2] << "m" << std::endl;
    }


    // Graph the final convex hull 
    graphPoints(approxHull, displayImage, minDepth, true);
    //graphPoints(finalHull, displayImage, minDepth, true);


    std::cout << std::endl;
}

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


void projectPoints(const std::vector< std::array<float, 7> >& hull, std::vector<cv::Point2f>& projectedPoints) {

    // Project the 3D points onto the 2D image plane
    for (const std::array<float, 7>& point : hull) {
        // TODO: try subsituting point[2] with minDepth to get better results
        float x = point[3] * (point[0] / point[2]) + point[5];
        float y = point[4] * (point[1] / point[2]) + point[6];

        projectedPoints.push_back(cv::Point2f(x, y));
    }

}
