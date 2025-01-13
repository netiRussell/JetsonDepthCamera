#include <iostream>
#include <depthai/depthai.hpp>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <vector>
#include <limits>

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
    int minDepth = 100;  // 0.1 meters
    int maxDepth = 450; // 0.4 meters

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

/// Generating a final convex hull with as little vertices as possible from the gathered points
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

    // Compute the convex hull of the projected points and store vertices to visualize them (optional)
    std::vector<cv::Point2f> finalHull;
    cv::convexHull(projectedPoints, finalHull);
    
    // Compute the convex hull of the projected points and store indices of vertices (to get original coordinates back later)
    std::vector<int> hullIndices;
    cv::convexHull(projectedPoints, hullIndices, false, false);

    // Get the original coordinates of the convex hull vertices
    std::vector<std::array<float, 7>> finalCoordinates;
    for (int idx : hullIndices) {
        gatheredPoints[idx][2] = minDepth;
        finalCoordinates.push_back(gatheredPoints[idx]); // Includes X, Y, Z, fx, fy, cx, cy
    }

    // Approximate the convex hull to get less vetices => better perfomance
    double arc_length = cv::arcLength(finalHull, true);
    std::vector<cv::Point2f> approxHull;
    std::vector<int> origIndices;
    cv::approxPolyDPWithIndices(finalHull, approxHull, origIndices, 0.01*arc_length, true); // epsilon 1% to 5% of the arc length

    // Getting the original version of the approximated coordinates 
    std::vector<std::array<float, 7>> approxCoordinates;
    for (int idx : origIndices){
        approxCoordinates.push_back(finalCoordinates[idx]);
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

    // The procedure
    for (const std::array<float, 7>& point : hull) {
        // TODO: try subsituting point[2] with minDepth to get better results
        float x = point[3] * (point[0] / point[2]) + point[5];
        float y = point[4] * (point[1] / point[2]) + point[6];

        projectedPoints.push_back(cv::Point2f(x, y));
    }

}

/// Helper to compute the perpendicular distance of point P to the line A->B
static double pointLineDistance(const cv::Point2f &P,
                                const cv::Point2f &A,
                                const cv::Point2f &B)
{
    // If A and B are the same point, just return distance from P to A
    double denom = cv::norm(B - A);
    if (denom < 1e-12) {
        return cv::norm(P - A);
    }

    // Vector AB
    cv::Point2f AB = B - A;
    // Vector AP
    cv::Point2f AP = P - A;

    // Projection length of AP onto AB, normalized by |AB|
    double t = (AP.dot(AB)) / (AB.dot(AB));

    // Closest point on AB to P
    // NOTE: If we want an unbounded line, we let t be any real number;
    // for a line segment, we could clamp t into [0, 1]. 
    // But standard RDP treats the line as infinite for distance measure.
    cv::Point2f proj = A + t * AB;

    return cv::norm(P - proj);
}

/// Recursive RDP function on vector of (Point2f, originalIndex)
static void rdp(const std::vector<std::pair<cv::Point2f,int>> &points,
                double epsilon,
                std::vector<std::pair<cv::Point2f,int>> &out)
{
    // If there are not enough points to simplify, just return them
    if (points.size() < 2) {
        out.insert(out.end(), points.begin(), points.end());
        return;
    }

    // Line from first to last point
    const cv::Point2f &start = points.front().first;
    const cv::Point2f &end   = points.back().first;

    // Find point with the maximum distance to the line
    double maxDist = -1.0;
    size_t index   = 0;

    for (size_t i = 1; i < points.size() - 1; i++) {
        double dist = pointLineDistance(points[i].first, start, end);
        if (dist > maxDist) {
            maxDist = dist;
            index   = i;
        }
    }

    // If max distance is greater than epsilon, recursively simplify
    if (maxDist > epsilon) {
        // RDP on the sub-vectors [0..index], [index..end]
        std::vector<std::pair<cv::Point2f,int>> leftIn(points.begin(), points.begin() + index + 1);
        std::vector<std::pair<cv::Point2f,int>> rightIn(points.begin() + index, points.end());

        std::vector<std::pair<cv::Point2f,int>> leftOut, rightOut;
        rdp(leftIn, epsilon, leftOut);
        rdp(rightIn, epsilon, rightOut);

        // The last point of leftOut is the same as the first point of rightOut,
        // so we can merge them carefully to avoid duplication.
        out.insert(out.end(), leftOut.begin(), leftOut.end() - 1);
        out.insert(out.end(), rightOut.begin(), rightOut.end());
    }
    else {
        // Just keep the endpoints
        out.push_back(points.front());
        out.push_back(points.back());
    }
}

/**
 * @param src     Input polygon points (e.g., your convex hull).
 * @param dst     Output approximated polygon points.
 * @param indices Output indices (relative to the original `src` vector).
 * @param epsilon Distance threshold for RDP.
 * @param closed  Whether the polygon is closed (like approxPolyDP's "closed" flag).
 */
void approxPolyDPWithIndices(const std::vector<cv::Point2f> &src, std::vector<cv::Point2f> &dst, std::vector<int> &indices, double epsilon, bool closed)
{
    dst.clear();
    indices.clear();

    if (src.empty()) {
        return;
    }

    // 1) Build a vector of (point, originalIndex).
    //    If the polygon is “closed”, we can replicate the first point at end or handle continuity.
    //    This example treats “closed” by connecting last -> first during the distance check.
    std::vector<std::pair<cv::Point2f,int>> points;
    points.reserve(src.size() + (closed ? 1 : 0));

    for (int i = 0; i < (int)src.size(); i++) {
        points.emplace_back(src[i], i);
    }

    // If closed, duplicate the first point at the end with the same index 
    // so the line end -> start is considered in RDP:
    if (closed) {
        points.emplace_back(src[0], 0);
    }

    // 2) Apply RDP on this vector
    std::vector<std::pair<cv::Point2f,int>> simplified;
    rdp(points, epsilon, simplified);

    // For closed shapes, after RDP, we might have the first point repeated at the end.
    // Typically, we remove the last repeated point.  Check if indices match or if the points match:
    if (closed && simplified.size() > 1) {
        if ( (simplified.front().second == simplified.back().second) ||
             (cv::norm(simplified.front().first - simplified.back().first) < 1e-7) )
        {
            simplified.pop_back();
        }
    }

    // 3) Copy the results to the output
    dst.reserve(simplified.size());
    indices.reserve(simplified.size());
    for (auto &p : simplified) {
        dst.push_back(p.first);
        indices.push_back(p.second);
    }
}

