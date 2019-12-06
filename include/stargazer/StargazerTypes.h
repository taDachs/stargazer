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

#include <map>
#include <vector>

namespace stargazer {

/**
 * @brief Definition of the six pose parameters. The rotation angles are given as rodriguez angles
 *
 */
enum struct POSE { X, Y, Z, Rx, Ry, Rz, N_PARAMS };

/**
 * @brief Definition of the intrinsic camera parameters
 *
 */
enum struct INTRINSICS { fu, fv, u0, v0, N_PARAMS };

/**
 * @brief Definition of the three position parmaters of a point
 *
 */
enum struct POINT { X, Y, Z, N_PARAMS };

/**
 * @brief A point is a 3D translation-only position. See ::POINT for the indexing scheme.
 */
typedef std::array<double, (int)POINT::N_PARAMS> Point;

/**
 * @brief This object hold the camera parameters. See ::INTRINSICS for the indexing scheme.
 */
typedef std::array<double, (int)INTRINSICS::N_PARAMS> camera_params_t;

/**
 * @brief This object hold the parameters of a translation and orientation pose. See ::POSE for the indexing scheme.
 */
typedef std::array<double, (int)POSE::N_PARAMS> pose_t;

/**
 * @brief Point generator function for a given ID.
 *
 * @param ID Landmark ID
 * @return std::vector<Point> List of points in landmark coordinates. The first three are the three corner points.
 */
std::vector<Point> getLandmarkPoints(int ID);  // Forward declaration

/**
 * @brief This class resembles a map landmark. After construction with the id,
 * the landmark holds its marker points in landmark coordinates.
 */
struct Landmark {
  ///--------------------------------------------------------------------------------------///
  /// The Landmarks are made similar to those from Hagisonic.
  /// ID of a landmark is coded see http://hagisonic.com/ for information on
  /// pattern
  ///--------------------------------------------------------------------------------------///

  /*  Numbering of corners and coordinate frame
   *  The origin of the landmark lies within Corner 1.
   *       ---> y'
   *  |   o   .   .
   *  |   .   .   .   .
   *  V   .   .   .   .
   *  x'  o   .   .   o
   *
   * The id of points (bit position)
   *          4   8
   *      1   5   9   13
   *      2   6   10  14
   *          7   11
   */

  Landmark() {}

  /**
   * @brief Constructor
   *
   * @param ID
   */
  Landmark(int ID) : id(ID), points(getLandmarkPoints(ID)) {}

  int id; /**< The landmarks id */
  std::array<double, static_cast<int>(POSE::N_PARAMS)> pose = {{0., 0., 0., 0., 0., 0.}}; /**< The landmarks pose */
  std::vector<Point> points; /**< Vector of landmark points. The first three are the corners */
  static constexpr int kGridCount = 4; /**< Number of rows and columns of a landmark */
  static constexpr double kGridDistance =
      0.08; /**< Distance between two adjacent landmark LEDs in meters */
};

inline std::vector<Point> getLandmarkPoints(const int ID) {
  std::vector<Point> points;

  // Add corner points
  const double lc(Landmark::kGridCount - 1);
  points.push_back({0., 0., 0.});
  points.push_back({lc, 0., 0.});
  points.push_back({lc, lc, 0.});

  // Add ID points
  for (int y = 0; y < Landmark::kGridCount; y++)  // For every column
  {
    for (int x = 0; x < Landmark::kGridCount; x++) {  // For every row
      /* StarLandmark IDs are coded: (here 4x4 grid)
       * x steps are binary shifts by 1
       * y steps are binary shifts by 4
       */
      if ((ID >> (Landmark::kGridCount * y + x)) & 1) {
        points.push_back({static_cast<double>(x), static_cast<double>(y), 0.});
      }
    }
  }
  // Apply landmark scale
  for (Point& p : points) {
    p[static_cast<int>(POINT::X)] *= Landmark::kGridDistance;
    p[static_cast<int>(POINT::Y)] *= Landmark::kGridDistance;
  }
  return points;
}

/**
 * @brief This class resembles the map representation. It holds a map of all known landmarks.
 */
typedef std::map<int, Landmark> landmark_map_t;

}  // namespace stargazer
