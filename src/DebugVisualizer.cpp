//
// This file is part of the stargazer library.
//
// Copyright 2016 Claudio Bandera <claudio.bandera@kit.edu (Karlsruhe Institute of Technology)
//
// The stargazer library is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// The stargazer library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <http://www.gnu.org/licenses/>.

#include "DebugVisualizer.h"

#include <iomanip>

#include <opencv2/imgproc.hpp>
#include <opencv2/viz/types.hpp>

#include "CoordinateTransformations.h"

using namespace stargazer;

const cv::Scalar DebugVisualizer::FZI_BLUE(163, 101, 0);
const cv::Scalar DebugVisualizer::FZI_GREEN(73, 119, 0);
const cv::Scalar DebugVisualizer::FZI_RED(39, 157, 236);

const int DebugVisualizer::POINT_RADIUS_IMG(1);
const int DebugVisualizer::POINT_RADIUS_MAP(4);
const int DebugVisualizer::POINT_THICKNESS(2);
const int DebugVisualizer::TEXT_OFFSET(25);
const double DebugVisualizer::FONT_SCALE(0.4);

void DebugVisualizer::prepareImg(cv::Mat& img) {
  if (img.type() == CV_8UC1) {
    // input image is grayscale
    cvtColor(img, img, CV_GRAY2RGB);
  }
}

void DebugVisualizer::ShowImage(const cv::Mat& img, std::string name) {
  cv::namedWindow(name, m_window_mode);
  cv::imshow(name, img);
  cv::waitKey(m_wait_time);
}

cv::Mat DebugVisualizer::DrawPoints(const cv::Mat& img, const std::vector<cv::Point> points) {
  const int marker_size(8);
  const int thickness(1);
  cv::Mat temp = img.clone();
  prepareImg(temp);
  for (auto& point : points) {
    cv::drawMarker(temp, point, FZI_GREEN, cv::MARKER_CROSS, marker_size, thickness);
  }
  return temp;
}

cv::Mat DebugVisualizer::DrawClusters(const cv::Mat& img,
                                      const std::vector<std::vector<cv::Point>> points) {
  cv::Mat temp = img.clone();
  prepareImg(temp);
  for (auto& group : points) {
    cv::Point median(0, 0);
    for (auto& point : group) {
      median += point;
      circle(temp, point, POINT_RADIUS_IMG, FZI_GREEN, POINT_THICKNESS);
    }
    median *= 1.0 / group.size();
    double variance = 0.0;
    for (auto& point : group) {
      variance += std::pow(median.x - point.x, 2) + std::pow(median.y - point.y, 2);
    }
    variance /= (group.size());
    int radius = static_cast<int>(2 * sqrt(variance));

    circle(temp, median, radius, FZI_BLUE, 2);
  }
  return temp;
}

cv::Mat DebugVisualizer::DrawLandmarkHypotheses(const cv::Mat& img,
                                                const std::vector<ImgLandmark>& landmarks) {
  cv::Mat temp = img.clone();
  prepareImg(temp);
  for (auto& lm : landmarks) {
    // Secants
    line(temp, lm.voCorners[1], lm.voCorners[0], cv::viz::Color::red());
    line(temp, lm.voCorners[1], lm.voCorners[2], cv::viz::Color::red());
    // Corners
    cv::drawMarker(temp, lm.voCorners[0], cv::viz::Color::red(), cv::MARKER_CROSS, 8, 2);  // leading corner clockwise (if assumption of rhs is valid)
    circle(temp, lm.voCorners[1], 3, cv::viz::Color::red(), POINT_THICKNESS);  // middle corner
    circle(temp, lm.voCorners[2], 3, cv::viz::Color::red(), POINT_THICKNESS);  // following corner clockwise
    // Inner points
    for (auto& imgPoint : lm.voIDPoints) {
      circle(temp, imgPoint, 1, FZI_GREEN, POINT_THICKNESS);
    }
    cv::Point median{(lm.voCorners[2].x + lm.voCorners[0].x) / 2,
                     (lm.voCorners[2].y + lm.voCorners[0].y) / 2};
    double radius = sqrt(pow(lm.voCorners[2].x - lm.voCorners[0].x, 2) +
                         pow(lm.voCorners[2].y - lm.voCorners[0].y, 2));
    circle(temp, median, radius, FZI_BLUE, 2);

    // Landmarks have no ID yet
  }
  return temp;
}

cv::Mat DebugVisualizer::DrawLandmarks(const cv::Mat& img,
                                       const std::vector<ImgLandmark>& landmarks) {
  cv::Mat temp = img.clone();
  prepareImg(temp);
  for (auto& lm : landmarks) {
    for (auto& imgPoint : lm.voCorners) {
      circle(temp, imgPoint, POINT_RADIUS_IMG, FZI_GREEN, POINT_THICKNESS);
    }
    for (auto& imgPoint : lm.voIDPoints) {
      circle(temp, imgPoint, POINT_RADIUS_IMG, FZI_GREEN, POINT_THICKNESS);
    }
    cv::Point median{(lm.voCorners[2].x + lm.voCorners[0].x) / 2,
                     (lm.voCorners[2].y + lm.voCorners[0].y) / 2};
    double radius = sqrt(pow(lm.voCorners[2].x - lm.voCorners[0].x, 2) +
                         pow(lm.voCorners[2].y - lm.voCorners[0].y, 2));
    circle(temp, median, radius, FZI_BLUE, 2);

    cv::Point imgPoint = lm.voCorners.front();
    imgPoint.x += TEXT_OFFSET;
    imgPoint.y += TEXT_OFFSET;
    putText(temp,
            getIDstring(lm.nID),
            imgPoint,
            cv::FONT_HERSHEY_DUPLEX,
            FONT_SCALE,
            cv::viz::Color::black());
  }
  return temp;
}

cv::Mat DebugVisualizer::DrawLandmarks(const cv::Mat& img,
                                       const landmark_map_t& landmarks,
                                       const camera_params_t& camera_intrinsics,
                                       const pose_t& ego_pose) {
  cv::Mat temp = img.clone();
  prepareImg(temp);
  cv::Point imgPoint;
  for (auto& lm : landmarks) {
    for (auto& pt : lm.second.points) {
      // Convert point into camera frame
      transformWorldToImgCv(pt, camera_intrinsics, ego_pose, imgPoint);
      circle(temp, imgPoint, POINT_RADIUS_MAP, FZI_RED, POINT_THICKNESS);
    }

    transformWorldToImgCv(lm.second.points.front(), camera_intrinsics, ego_pose, imgPoint);
    imgPoint.x += TEXT_OFFSET;
    imgPoint.y += TEXT_OFFSET - 28. * FONT_SCALE;
    putText(temp,
            getIDstring(lm.second.id),
            imgPoint,
            cv::FONT_HERSHEY_DUPLEX,
            FONT_SCALE,
            cv::viz::Color::black());
  }
  return temp;
}

void DebugVisualizer::transformWorldToImgCv(const Point& p,
                                            const camera_params_t& camera_intrinsics,
                                            const pose_t& ego_pose,
                                            cv::Point& p_img) {
  double x, y;
  transformWorldToImg(p[static_cast<int>(POINT::X)],
                      p[static_cast<int>(POINT::Y)],
                      p[static_cast<int>(POINT::Z)],
                      ego_pose.data(),
                      camera_intrinsics.data(),
                      &x,
                      &y);
  p_img.x = static_cast<int>(x);
  p_img.y = static_cast<int>(y);
}

std::string DebugVisualizer::getIDstring(int id) {
  std::stringstream textstream;
  textstream << "ID: " << std::showbase << std::internal << std::setfill('0')
             << std::setw(6) << std::hex << id;
  return textstream.str();
}
