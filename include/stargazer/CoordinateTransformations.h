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

#include <ceres/rotation.h>

#include <iostream>

#include "StargazerTypes.h"

namespace stargazer {

/**
 * @brief This function will transform a point, given in landmark coordinates into world coordinates
 *
 * @param x_landmark  x value of input point in landmark coordinates
 * @param y_landmark  y value of input point in landmark coordinates
 * @param landmark_position the position of the landmark
 * @param landmark_orientation the orientation of the landmark (as quaternion)
 * @param x_world x value of ouput point in world coordinates
 * @param y_world y value of ouput point in world coordinates
 * @param z_world z value of ouput point in world coordinates
 */
template <typename T>
void transformLandMarkToWorld(const T& x_landmark,
                              const T& y_landmark,
                              const T* const landmark_position,
                              const T* const landmark_orientation,
                              T* const x_world,
                              T* const y_world,
                              T* const z_world) {
  // Create point in Landmark coordinates
  const T point_landmark[3] = {x_landmark, y_landmark, T(0.0)};

  // Transform point from Landmark to world coordinates
  T point_world[3];
  ceres::UnitQuaternionRotatePoint(landmark_orientation, point_landmark, point_world);

  // lm_pose[0,1,2] are the translation.
  point_world[0] += landmark_position[(int)POINT::X];
  point_world[1] += landmark_position[(int)POINT::Y];
  point_world[2] += landmark_position[(int)POINT::Z];

  *x_world = point_world[0];
  *y_world = point_world[1];
  *z_world = point_world[2];
}


/**
 * @brief This function will transform a point, given in world coordinates into image coordinates
 *
 * @param x_world  x value of input point in world coordinates
 * @param y_world  y value of input point in world coordinates
 * @param z_world  z value of input point in world coordinates
 * @param camera_position the position of the camera
 * @param camera_orientation the orientation of the camera (as quaternion)
 * @param camera_intrinsics the cameras intrinsic parameters
 * @param x_image x value of ouput point in image coordinates
 * @param y_image y value of ouput point in image coordinates
 */
template <typename T>
void transformWorldToImg(const T& x_world,
                         const T& y_world,
                         const T& z_world,
                         const T* const camera_position,
                         const T* const camera_orientation,
                         const T* const camera_intrinsics,
                         T* const x_image,
                         T* const y_image) {

  T rel_camera[(int)POINT::N_PARAMS];
  rel_camera[0] = camera_position[(int)POINT::X] - x_world;
  rel_camera[1] = camera_position[(int)POINT::Y] - y_world;
  rel_camera[2] = camera_position[(int)POINT::Z] - z_world;

  // Inverse quaternion
  T inv_quat[(int)QUAT::N_PARAMS];
  inv_quat[(int)QUAT::W] = camera_orientation[(int)QUAT::W];
  inv_quat[(int)QUAT::X] = -camera_orientation[(int)QUAT::X];
  inv_quat[(int)QUAT::Y] = -camera_orientation[(int)QUAT::Y];
  inv_quat[(int)QUAT::Z] = -camera_orientation[(int)QUAT::Z];

  // Transform point to camera coordinates
  T p_camera[(int)POINT::N_PARAMS];
  ceres::UnitQuaternionRotatePoint(inv_quat, rel_camera, p_camera);

  // Transform point to image coordinates
  T p_image[(int)POINT::N_PARAMS];
  p_image[0] = camera_intrinsics[(int)INTRINSICS::fu] * p_camera[0] +
               camera_intrinsics[(int)INTRINSICS::u0] * p_camera[2];
  p_image[1] = camera_intrinsics[(int)INTRINSICS::fv] * p_camera[1] +
               camera_intrinsics[(int)INTRINSICS::v0] * p_camera[2];
  p_image[2] = p_camera[2];
  if (p_image[2] == T(0)) {
    std::cout << "WARNING; Attempt to divide by 0!" << std::endl;
    return;
  }
  p_image[0] = p_image[0] / p_image[2];
  p_image[1] = p_image[1] / p_image[2];
  p_image[2] = p_image[2] / p_image[2];

  *x_image = p_image[0];
  *y_image = p_image[1];
}

/**
 * @brief This function will transform a point, given in landmark coordinates into image coordinates
 *
 * @param x_landmark  x value of input point in landmark coordinates
 * @param y_landmark  y value of input point in landmark coordinates
 * @param landmark_position the position of the landmark
 * @param landmark_orientation the orientation of the landmark (as quaternion)
 * @param camera_position the position of the camera
 * @param camera_orientation the orientation of the camera (as quaternion)
 * @param camera_intrinsics the cameras intrinsic parameters
 * @param x_image x value of ouput point in image coordinates
 * @param y_image y value of ouput point in image coordinates
 */
template <typename T>
void transformLandMarkToImage(const T& x_landmark,
                              const T& y_landmark,
                              const T* const landmark_position,
                              const T* const landmark_orientation,
                              const T* const camera_position,
                              const T* const camera_orientation,
                              const T* const camera_intrinsics,
                              T* const x_image,
                              T* const y_image) {
  T x_world, y_world, z_world;
  transformLandMarkToWorld(
      x_landmark, y_landmark, landmark_position, landmark_orientation, &x_world, &y_world, &z_world);
  transformWorldToImg(
      x_world, y_world, z_world, camera_position, camera_orientation, camera_intrinsics, x_image, y_image);
}

}  // namespace stargazer
