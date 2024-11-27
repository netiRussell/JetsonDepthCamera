#include <iostream>
#include <depthai/depthai.hpp>
#include <opencv2/opencv.hpp>
#include <cmath>

ushort interpolateDepth(const cv::Mat& depthImage, int x, int y);
void analyzeCaptures( const std::vector< std::array<float, 5> >& gatheredPoints, double minDepth );
cv::Point2f projectPoint(const cv::Point3f& point, float focalLength, const cv::Point2f& center);
void graphPoints( std::vector< std::array<float, 5> > hull );
std::vector<cv::Point2f> transformPoints(const std::vector<std::array<float, 5>>& points);

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
    int minDepth = 500;  // 0.5 meters
    int maxDepth = 800; // 0.8 meters

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
	cv::minMaxLoc(mask, &minDepthInMask, nullptr, nullptr, nullptr);

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

        // Create an image to display
        cv::Mat displayImage;
        cv::normalize(depthImage, displayImage, 255, 0, cv::NORM_INF, CV_8UC1);
        cv::equalizeHist(displayImage, displayImage);
        cv::applyColorMap(displayImage, displayImage, cv::COLORMAP_HOT);

        // Draw convex hulls on the image
        for (size_t i = 0; i < hulls.size(); i++) {
            cv::drawContours(displayImage, hulls, static_cast<int>(i), cv::Scalar(0, 255, 0), 2);
        }

        // Filter out noise
        for (size_t i = 0; i < hulls.size(); i++) {

            std::vector< std::array<float, 5> > points; // Points of a single convex hull

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

        // Display image
        cv::imshow("Convex Hulls", displayImage);

        // Analyze and clear the hulls that don't represent real objects
        if( i == nCaptPerAnalysis-1 ){

            // Check if the hull represents a real object
            for (size_t j = 0; j < netHulls.size(); j++) {

                // Check on other hulls
                std::cout << "Current Area = " << netHulls[j].area << std::endl;
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

                        std::cout << "\t Considered Area = " << netHulls[k].area << std::endl;
                        std::cout << "Hull " << j << ": Centroid = (" << netHulls[j].coordinates.first << ", " << netHulls[j].coordinates.second << ")" << std::endl;
                    }

                }

            }


            // ! TODO: change the structure to have all of the coordinates in a single capture
            // Filter out coordinates for visualization
            std::vector< std::array<float, 5> > points;

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

                    points.push_back({X, Y, Z, fx, fy});
                }

            }

            analyzeCaptures(points, minDepthInMask);

            // To restart the loop:
            // i = 1;
            // Clear the netHulls vector
        }

        i++;
    }

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


void analyzeCaptures( const std::vector< std::array<float, 5> >& gatheredPoints, double minDepth){
    // Shape of gatheredCaptures = # of captures, # of convex hulls, # of coordinates, 5 coordinatex - X, Y, Z, cx, cy.
    std::cout << "\n------------------------------------------------------------------------\n";

    // Main loop of the function
    std::cout << "\tCurrent Convex Hull size = " << gatheredPoints.size() << "\n";

    for( const std::array<float, 5> &coordinates : gatheredPoints ){
            std::cout << "\t\tPoint: X=" << coordinates[0] << "m, Y=" << coordinates[1] << "m, Z=" << coordinates[2] << "m" << std::endl;
    }

    std::cout << "The minimal depth of the object: " << minDepth;

    // Compute the final convex hull
    std::vector<cv::Point> finalHull(gatheredPoints.size());
    std::vector<cv::Point2f> points2d = transformPoints(gatheredPoints);
    cv::convexHull(points2d, finalHull);

    graphPoints(finalHull, minDepth);

    std::cout << std::endl;
}


void graphPoints( std::vector<cv::Point> hull, double minDepth ){
    // Set up the display window and projection parameters
    int width = 1280, height = 720;
    cv::Mat image = cv::Mat::zeros(height, width, CV_8UC3);
    cv::Point2f center(width / 2, height / 2);  // Center of the 2D plane

    // Draw each 3D point on the 2D image
    for (const cv::Point point : hull) { 
        // Project the 3D point onto the 2D image plane
        cv::Point2f pt2D = cv::Point2f(point[0] + center.x, point[1] + center.y);

        // Scale the circle size based on the Z coordinate to simulate depth
        int radius = static_cast<int>(10 / minDepth);  // Adjust size based on depth

        // Draw the projected point as a circle on the 2D plane
        cv::circle(image, pt2D, radius, (0, 255, 255), -1);  // -1 fills the circle

        // Put the text on the image
        std::string text = std::to_string(point[0]) + ", " + std::to_string(point[1]) + ", " + std::to_string(minDepth);
        cv::putText(image, text, pt2D, cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(0, 0, 255), 0.75);
    }

    // Display the result
    cv::imshow("3D Points Projection", image);
    cv::waitKey(0);
}

std::vector<cv::Point2f> transformPoints(const std::vector<std::array<float, 5>>& points) {
    std::vector<cv::Point2f> result;
    result.reserve(points.size()); // Reserve space for efficiency.
    
    for (const auto& point : points) {
        result.emplace_back(point[0], point[1]); // Create cv::Point2f with X and Y.
    }
    
    return result;
}

/* terminate called after throwing an instance of 'cv::Exception'
  what():  OpenCV(4.5.4) /home/ubuntu/build_opencv/opencv/modules/core/src/matrix_wrap.cpp:1393: error: (-215:Assertion failed) mtype == type0 || (CV_MAT_CN(mtype) == CV_MAT_CN(type0) && ((1 << type0) & fixedDepthMask) != 0) in function 'create'
*/ 
