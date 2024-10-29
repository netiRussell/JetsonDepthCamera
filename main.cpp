#include <iostream>
#include <depthai/depthai.hpp>
#include <opencv2/opencv.hpp>

ushort interpolateDepth(const cv::Mat& depthImage, int x, int y);
void printCaptures( const std::vector< std::vector< std::vector< std::array<float, 3> > > >& gatheredCaptures  );
cv::Point2f projectPoint(const cv::Point3f& point, float focalLength, const cv::Point2f& center);
void graphPoints( std::vector< std::array<float, 3> > hull );

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

    // Get depth frame
    auto depthFrame = depthQueue->get<dai::ImgFrame>();
    cv::Mat depthImage = depthFrame->getFrame();
    // Get the resolution
    int width = depthImage.cols;
    int height = depthImage.rows;
    
    auto intrinsics = calibData.getCameraIntrinsics(dai::CameraBoardSocket::RIGHT, width, height, true);

    // Extract intrinsic parameters
    float fx = intrinsics[0][0];
    float fy = intrinsics[1][1];
    float cx = intrinsics[0][2];
    float cy = intrinsics[1][2];

    // Depth thresholds in millimeters
    int minDepth = 500;  // 0.5 meters
    int maxDepth = 800; //

    int i = 1;
    /*
     * gatheredCaptures - vector that holds all the captures from the while loop.
     * Each Capture holds all convex hulls found in the capture
     * Each Hull holds its coordinates calculated in the most inner for-loop
     *
     * Shape = # of captures, # of convex hulls, # of coordinates, 3 coordinatex - X, Y, Z.
    */
    std::vector< std::vector< std::vector< std::array<float, 3> > > > gatheredCaptures;
    while (i <= 20) {
        std::cout << "Capture #" << i << "\n";

        // Get depth frame
        auto depthFrame = depthQueue->get<dai::ImgFrame>();
        cv::Mat depthImage = depthFrame->getFrame();

        // Threshold depth image to create mask
        cv::Mat mask;
        cv::inRange(depthImage, minDepth, maxDepth, mask);


        /* AT THIS STEP THE DEPTH IS CORRECT*/


        // Noise reduction
        cv::erode(mask, mask, cv::Mat(), cv::Point(-1, -1), 2);
        cv::dilate(mask, mask, cv::Mat(), cv::Point(-1, -1), 2);

        // Blur
        cv::GaussianBlur(mask, mask, cv::Size(9, 9), 5);


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
        cv::normalize(depthImage, displayImage, 0, 255, cv::NORM_MINMAX, CV_8UC1);
        cv::cvtColor(displayImage, displayImage, cv::COLOR_GRAY2BGR);

        // Draw convex hulls on the image
        for (size_t i = 0; i < hulls.size(); i++) {
            cv::drawContours(displayImage, hulls, static_cast<int>(i), cv::Scalar(0, 255, 0), 2);
        }


        // For each convex hull, compute and print 3D coordinates
        std::vector< std::vector< std::array<float, 3> > > gatheredHulls; // Vector that stores hulls of current capture

        for (size_t i = 0; i < hulls.size(); i++) {

            std::cout << "Convex Hull " << i << ":" << std::endl;
            std::vector< std::array<float, 3> > points; // Points of a single convex hull

            for (const auto& point : hulls[i]) {
                int x = point.x;
                int y = point.y;

                ushort depthValue = depthImage.at<ushort>(y, x);

                if (depthValue == 0){
                    // Try to interpolate depth from neighboring pixels
                    depthValue = interpolateDepth(depthImage, x, y);

                    if (depthValue == 0)
                        continue; // If still zero, skip the point
                }
                
                // Adjust depth value based on extended disparity
                float adjustedDepthValue = static_cast<float>(depthValue) / 2.0f; // Divide by 2 due to extended disparity

                float Z = adjustedDepthValue / 1000.0f; // Convert mm to meters
                float X = (x - cx) * Z / fx;
                float Y = (y - cy) * Z / fy;

                points.push_back({X, Y, Z});

                // Output coordinates
                //std::cout << "Point: X=" << X << "m, Y=" << Y << "m, Z=" << Z << "m" << std::endl;
            }

            gatheredHulls.push_back(points);
        }

        gatheredCaptures.push_back(gatheredHulls);

        // Display image
        cv::imshow("Convex Hulls", displayImage);

        // Exit loop if 'q' is pressed
        if (cv::waitKey(1) == 'q') {
            break;
        }

        i++;
    }

    // Print Captures
    printCaptures( gatheredCaptures );

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


void printCaptures( const std::vector< std::vector< std::vector< std::array<float, 3> > > >& gatheredCaptures  ){
    // Shape of gatheredCaptures = # of captures, # of convex hulls, # of coordinates, 3 coordinatex - X, Y, Z.
    std::cout << "\n------------------------------------------------------------------------\n";

    for( int i = 0; i < gatheredCaptures.size(); i++ ){
        std::cout << "Capture #" << i << " | Size = " << gatheredCaptures[i].size() << "\n";

        for( std::vector< std::array<float, 3> > hull : gatheredCaptures[i] ){
            std::cout << "\tCurrent Convex Hull size = " << hull.size() << "\n";

            for( std::array<float, 3> coordinates : hull ){
                std::cout << "\t\tPoint: X=" << coordinates[0] << "m, Y=" << coordinates[1] << "m, Z=" << coordinates[2] << "m" << std::endl;
            }

            graphPoints(hull);
            std::cout << "\n";
        }

        std::cout << "\n\n";
    }

    std::cout << std::endl;
}


cv::Point2f projectPoint(const std::array<float, 3>& point, float focalLength, const cv::Point2f& center) {
    float x = focalLength * (point[0] / point[2]) + center.x;
    float y = focalLength * (point[1] / point[2]) + center.y;
    return cv::Point2f(x, y);
}


void graphPoints( std::vector< std::array<float, 3> > hull ){
    // Set up the display window and projection parameters
    int width = 600, height = 400;
    cv::Mat image = cv::Mat::zeros(height, width, CV_8UC3);
    float focalLength = 500.0;  // Adjust focal length to control "zoom"
    cv::Point2f center(width / 2, height / 2);  // Center of the 2D plane

    // Draw each 3D point on the 2D image
    for (const std::array<float, 3>& point : hull) {
        // Project the 3D point onto the 2D image plane
        cv::Point2f pt2D = projectPoint(point, focalLength, center);

        // Scale the circle size based on the Z coordinate to simulate depth
        int radius = static_cast<int>(10 / point[2]);  // Adjust size based on depth
        cv::Scalar color(0, 255 - static_cast<int>(100 * (1.0 / point[2])), 255);  // Color changes with Z

        // Draw the projected point as a circle on the 2D plane
        cv::circle(image, pt2D, radius, color, -1);  // -1 fills the circle
    }

    // Display the result
    cv::imshow("3D Points Projection", image);
    cv::waitKey(0);
}
