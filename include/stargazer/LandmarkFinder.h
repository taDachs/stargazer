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

#pragma once

#include <vector>

#include <opencv2/features2d.hpp>

#include "StargazerConfig.h"
#include "StargazerImgTypes.h"
#include "StargazerTypes.h"

namespace stargazer {

/**
 * @brief This class detects landmarks in images.
 */
class LandmarkFinder {
 public:
  /**
   * @brief Constructor.
   *
   * @param cfgfile Path to map file with landmark poses.
   * @remark The config file has to be generated with ::writeMapConfig!
   */
  LandmarkFinder(std::string cfgfile);

  /**
   * @brief Destructor
   */
  ~LandmarkFinder();

  /**
   * @brief Main worker function. Writes all detected landmarks into vector
   *
   * @param img Image to analyze
   * @param detected_landmarks Output vector of detected landmarks
   * @return int Error code
   */
  int DetectLandmarks(const cv::Mat& img, std::vector<ImgLandmark>& detected_landmarks);

  cv::Mat grayImage_;             /**< Keeps a copy of the grayvalue image */
  std::vector<cv::Point> points_; /**< Keeps a copy of points found */
  std::vector<Cluster> clusteredPoints_; /**< Keeps a copy of point clusters found*/
  std::vector<ImgLandmark> landmarkHypotheses_; /**< Keeps a copy of landmark hypotheses*/

  std::vector<uint16_t> valid_ids_; /**< Vector of valid IDs, read from map*/

  // parameters for point detection
  cv::SimpleBlobDetector::Params blobParams;

  // parameters for clustering
  double maxRadiusForCluster; /**< Maximum radius for clustering marker points to landmarks*/
  uint16_t minPointsPerLandmark; /**< Minimum count of marker points per landmark (0)*/
  uint16_t maxPointsPerLandmark; /**< Maximum count of marker points per landmark (depends on grid used)*/

  // parameters for corner hypotheses
  int maxCornerHypotheses; /**< Maximum number of corner hypotheses which are still considered*/
  double cornerHypothesesCutoff; /**< Defines near-best corner points hypotheses which are still considered further*/
  double cornerAngleTolerance;
  double pointInsideTolerance;
  double fwLengthTriangle; /**< Weight factor for the circumference of the triangle*/
  double fwCrossProduct; /**< Weight factor for cross product of the secants*/

  // parameters for id calculation
  double idPointThresholdBackwards; /**< Threshold for id points in image for backwards calculation*/

 private:
  static constexpr int DIM = 4;

  /**
   * @brief Uses SimpleBlobDetection for point detection
   *
   * @param img_in raw image
   */
  std::vector<cv::Point> FindBlobs(cv::Mat& img_in) const;

  /**
   * @brief Finds hypotheses for landmarks by clustering the input points
   *
   * @param points_in
   * @param clusters
   * @param radiusThreshold
   * @param minPointsThreshold
   * @param maxPointsThreshold
   */
  void FindClusters(const std::vector<cv::Point>& points_in,
                    std::vector<Cluster>& clusters,
                    const double radiusThreshold,
                    const unsigned int minPointsThreshold,
                    const unsigned int maxPointsThreshold) const;

  /**
   * @brief Finds hypotheses for the three corner points of a landmark given a
   * point cluster. It utilizes a score function to find good triples.
   *
   * @param points cluster of points
   * @return std::vector<ImgLandmark> reasonable hypotheses
   */
  std::vector<ImgLandmark> FindCorners(const std::vector<cv::Point>& points);

  /**
   * @brief Finds valid landmark observations from the input hypotheses
   *
   * @param clusteredPoints
   * @return std::vector<ImgLandmark>
   */
  std::vector<ImgLandmark> FindLandmarks(const std::vector<Cluster>& clusteredPoints);

  /**
   * @brief Tries to identify the landmarks ID
   *
   * @param landmarks vector of observations
   */
  void GetIDs(std::vector<ImgLandmark>& landmarks);

  /**
   * @brief Tries to calculate the landmarks id by transforming the observed
   * points into unary landmark coordinates.
   *
   * @param landmark
   * @return uint16_t calculated ID
   */
  uint16_t CalculateIdForward(const ImgLandmark& landmark) const;

  /**
   * @brief   Tryies to calculate the landmarks id by looking in the filtered
   * image, whether a bright point can be seen where it is assumed.
   *
   * @param landmark
   * @return bool
   */
  bool CalculateIdBackward(ImgLandmark& landmark) const;

  /**
   * @brief Transforms global point coordinates to local landmark coordinates
   *
   * @param x0y0 Global coordinates of corner (local x=0 y=0)
   * @param x1y0 Global coordinates of corner (local x=1 y=0)
   * @param x1y1 Global coordinates of corner (local x=1 y=1)
   * @param p vector of points to transform
   */
  void TransformToLocalPoints(const cv::Point2f& x0y0,
                              const cv::Point2f& x1y0,
                              const cv::Point2f& x1y1,
                              std::vector<cv::Point2f>& p) const;

  /**
   * @brief Transforms local point coordinates to global image coordinates
   *
   * @param x0y0 Global coordinates of corner (local x=0 y=0)
   * @param x1y0 Global coordinates of corner (local x=1 y=0)
   * @param x1y1 Global coordinates of corner (local x=1 y=1)
   * @param p vector of points to transform
   */
  void TransformToGlobalPoints(const cv::Point2f& x0y0,
                               const cv::Point2f& x1y0,
                               const cv::Point2f& x1y1,
                               std::vector<cv::Point2f>& p) const;

  template <typename T>
  inline bool isInside(T value, T lower, T upper, T tol) {
    return value > lower - tol && value < upper + tol;
  }
};

}  // namespace stargazer
