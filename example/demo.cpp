#include <CeresLocalizer.h>
#include <DebugVisualizer.h>

#include "LandmarkFinder.h"

using namespace stargazer;
using namespace std;

int main(int argc, char** argv) {
  if (argc != 3) {
    cout << " Usage: " << argv[0] << " <image_file> <config_file>" << endl;
    return -1;
  }
  cv::Mat input_image = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);  // Read the file

  DebugVisualizer debugVisualizer;
  debugVisualizer.SetWaitTime(-1);  // Wait until user has pressed key
  debugVisualizer.SetWindowMode(CV_WINDOW_NORMAL);

  LandmarkFinder landmarkFinder(argv[2]);
  std::vector<ImgLandmark> detected_landmarks;
  landmarkFinder.DetectLandmarks(input_image, detected_landmarks);

  cout << "Displaying images, press any key to continue.... " << endl;

  // Invert images for better visibilty
  cv::bitwise_not(landmarkFinder.grayImage_, landmarkFinder.grayImage_);
  debugVisualizer.ShowImage(landmarkFinder.grayImage_, "0 Gray Image");

  // Draw detections
  debugVisualizer.ShowImage(
      debugVisualizer.DrawPoints(landmarkFinder.grayImage_, landmarkFinder.points_),
      "1 Points");
  debugVisualizer.ShowImage(
      debugVisualizer.DrawClusters(landmarkFinder.grayImage_, landmarkFinder.clusteredPoints_),
      "2 Clusters");
  debugVisualizer.ShowImage(debugVisualizer.DrawLandmarkHypotheses(
                                landmarkFinder.grayImage_, landmarkFinder.landmarkHypotheses_),
                            "3 Hypotheses");
  debugVisualizer.ShowImage(
      debugVisualizer.DrawLandmarks(landmarkFinder.grayImage_, detected_landmarks),
      "4 Landmarks");

  // Localize
  const std::string args(argv[2]);
  CeresLocalizer localizer(args);
  localizer.UpdatePose(detected_landmarks, 0.0);
  cout << localizer.getSummary().FullReport() << endl << endl;

  Pose pose = localizer.getPose();
  // clang-format off
  cout << "Pose is"
       << " x=" << pose.position[(int)POINT::X]
       << " y=" << pose.position[(int)POINT::Y]
       << " z=" << pose.position[(int)POINT::Z]
       << " qw=" << pose.orientation[(int)QUAT::W]
       << " qx=" << pose.orientation[(int)QUAT::X]
       << " qy=" << pose.orientation[(int)QUAT::Y]
       << " qz=" << pose.orientation[(int)QUAT::Z] << endl;
  // clang-format on

  return EXIT_SUCCESS;
}
