/*
    Goal: Find coordinates of all the objects that are closer to the camera than the distance d

    Next step: Make some kind of a flag for an object to find just the coordinates of that object 
*/

#include <depthai/depthai.hpp>
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

int main() {
    // Set the threshold distance in meters
    float d = 2.0f;

    // Create pipeline
    dai::Pipeline pipeline;

    // Define sources and outputs
    auto monoLeft = pipeline.create<dai::node::MonoCamera>();
    auto monoRight = pipeline.create<dai::node::MonoCamera>();
    auto stereo = pipeline.create<dai::node::StereoDepth>();
    auto xoutDepth = pipeline.create<dai::node::XLinkOut>();

    xoutDepth->setStreamName("depth");

    // Properties
    monoLeft->setResolution(dai::MonoCameraProperties::SensorResolution::THE_400_P);
    monoLeft->setBoardSocket(dai::CameraBoardSocket::LEFT);
    monoRight->setResolution(dai::MonoCameraProperties::SensorResolution::THE_400_P);
    monoRight->setBoardSocket(dai::CameraBoardSocket::RIGHT);

    stereo->setDefaultProfilePreset(dai::node::StereoDepth::PresetMode::HIGH_DENSITY);
    stereo->setDepthAlign(dai::CameraBoardSocket::RGB);
    stereo->setOutputDepth(true);

    // Linking
    monoLeft->out.link(stereo->left);
    monoRight->out.link(stereo->right);
    stereo->depth.link(xoutDepth->input);

    // Initialize the device
    dai::Device device(pipeline);

    // Get calibration data
    dai::CalibrationHandler calibData = device.readCalibration();
    std::vector<std::vector<float>> intrinsics = calibData.getCameraIntrinsics(
        dai::CameraBoardSocket::LEFT, 640, 400);

    // Access fx, fy, cx, cy
    float fx = intrinsics[0][0];
    float fy = intrinsics[1][1];
    float cx = intrinsics[0][2];
    float cy = intrinsics[1][2];

    // Output queue to receive depth frames
    auto qDepth = device.getOutputQueue("depth", 4, false);

    // Convert threshold distance to millimeters
    float d_mm = d * 1000.0f; // Convert meters to millimeters

    while (true) {
        // Get the depth frame
        auto inDepth = qDepth->get<dai::ImgFrame>(); // Blocking call

        // Depth frame as cv::Mat (depthFrame is CV_16UC1)
        cv::Mat depthFrame = inDepth->getCvFrame();

        // Convert depth frame to float (CV_32F)
        cv::Mat depthFrameFloat;
        depthFrame.convertTo(depthFrameFloat, CV_32F);

        // Create a mask where depth is less than d_mm
        cv::Mat mask = depthFrameFloat < d_mm;

        // Find indices of points within the threshold distance
        std::vector<cv::Point> indices;
        cv::findNonZero(mask, indices);

        // Prepare vectors to store coordinates
        std::vector<float> X;
        std::vector<float> Y;
        std::vector<float> Z;
        X.reserve(indices.size());
        Y.reserve(indices.size());
        Z.reserve(indices.size());

        // Iterate over indices to compute real-world coordinates
        for (const auto& pt : indices) {
            int x = pt.x;
            int y = pt.y;

            float z = depthFrameFloat.at<float>(y, x); // Depth value in mm

            // Compute X, Y using the pinhole camera model
            float X_val = (x - cx) * z / fx;
            float Y_val = (y - cy) * z / fy;

            X.push_back(X_val / 1000.0f); // Convert to meters
            Y.push_back(Y_val / 1000.0f); // Convert to meters
            Z.push_back(z / 1000.0f);     // Convert to meters
        }

        // Combine X, Y, Z into coordinates
        std::vector<cv::Point3f> coordinates;
        coordinates.reserve(X.size());
        for (size_t i = 0; i < X.size(); ++i) {
            coordinates.emplace_back(X[i], Y[i], Z[i]);
        }

        // Output the number of points detected
        std::cout << "Number of points within " << d << " meters: " << coordinates.size() << std::endl;

        // Add any additional processing or visualization here

        // Break condition (optional)
        // if (some_condition)
        //     break;
    }

    return 0;
}
