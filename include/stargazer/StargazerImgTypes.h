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

#include <opencv2/core/types.hpp>

#include "StargazerTypes.h"

namespace stargazer {

/**
 * @brief A cluster is an unordered collection of points, it represents a hypothesis for a landmark.
 */
typedef std::vector<cv::Point> Cluster;

/**
 * @brief An image landmark holds the information of an observed landmark. All coordinates are in image coordinates.
 */
struct ImgLandmark {
  uint16_t nID;                   /**< The detected ID of the landmark */
  std::array<cv::Point, NUM_CORNERS> corners; /**< The three corners of the landmark */
  std::vector<cv::Point> idPoints; /**< The inner points of the landmark, which encode the ID */
};

/**
 * @brief Converts an ImgLandmrk to a Map Landmark
 *
 * @param lm_in
 * @return Landmark
 */
inline Landmark convert2Landmark(ImgLandmark& lm_in) {
  Landmark lm_out(lm_in.nID);
  lm_out.points.clear();

  for (auto& el : lm_in.corners) {
    Point pt = {static_cast<double>(el.x), static_cast<double>(el.y), 0.};
    lm_out.points.push_back(pt);
  }
  for (auto& el : lm_in.idPoints) {
    Point pt = {static_cast<double>(el.x), static_cast<double>(el.y), 0.};
    lm_out.points.push_back(pt);
  }

  return lm_out;
};

}  // namespace stargazer
