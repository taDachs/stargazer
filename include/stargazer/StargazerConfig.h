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

#include <yaml-cpp/yaml.h>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

#include "StargazerTypes.h"

namespace stargazer {

inline YAML::Node loadYaml(const std::string& cfgfile) {
  YAML::Node config;
  try {
    config = YAML::LoadFile(cfgfile);
  } catch (YAML::BadFile& /*e*/) {
    std::string msg = "Stargazer config file does not exist: " + cfgfile;
    throw std::runtime_error(msg);
  } catch (YAML::ParserException& /*e*/) {
    std::string msg = "Wrong YAML syntax in stargazer config file: " + cfgfile;
    throw std::runtime_error(msg);
  }
  return config;
}

/**
 * @brief
 *
 * @param cfgfile
 * @param camera_intrinsics
 */
inline void readCamConfig(const std::string& cfgfile, camera_params_t& camera_intrinsics) {
  YAML::Node config(loadYaml(cfgfile));

  if (config["CameraIntrinsics"]) {
    camera_intrinsics[(int)INTRINSICS::fu] =
        config["CameraIntrinsics"]["fu"].as<double>();
    camera_intrinsics[(int)INTRINSICS::fv] =
        config["CameraIntrinsics"]["fv"].as<double>();
    camera_intrinsics[(int)INTRINSICS::u0] =
        config["CameraIntrinsics"]["u0"].as<double>();
    camera_intrinsics[(int)INTRINSICS::v0] =
        config["CameraIntrinsics"]["v0"].as<double>();
  } else {
    std::string msg =
        "Stargazer camera config file is missing CameraIntrinics!: " + cfgfile;
    throw std::runtime_error(msg);
  }
}

/**
 * @brief
 *
 * @param cfgfile
 * @param landmarks
 */
inline void readMapConfig(const std::string& cfgfile, landmark_map_t& landmarks) {
  YAML::Node config(loadYaml(cfgfile));

  if (config["Landmarks"]) {
    for (size_t i = 0; i < config["Landmarks"].size(); i++) {
      auto lm = config["Landmarks"][i];

      int id = lm["HexID"].as<int>();
      landmarks[id] = Landmark(id);

      Pose pose;
      pose.position[(int)POINT::X] = lm["x"].as<double>();
      pose.position[(int)POINT::Y] = lm["y"].as<double>();
      pose.position[(int)POINT::Z] = lm["z"].as<double>();
      pose.orientation[(int)QUAT::W] = lm["qw"].as<double>();
      pose.orientation[(int)QUAT::X] = lm["qx"].as<double>();
      pose.orientation[(int)QUAT::Y] = lm["qy"].as<double>();
      pose.orientation[(int)QUAT::Z] = lm["qz"].as<double>();
      landmarks[id].pose = pose;
    }
  } else {
    std::string msg = "Stargazer map config file is missing Landmarks!: " + cfgfile;
    throw std::runtime_error(msg);
  }
}

/**
 * @brief
 *
 * @param cfgfile
 * @param camera_intrinsics
 */
inline void writeCamConfig(const std::string& cfgfile, const camera_params_t& camera_intrinsics) {
  std::ofstream fout(cfgfile);

  fout << "CameraIntrinsics:\n";
  fout << " fu: " << camera_intrinsics[(int)INTRINSICS::fu] << "\n";
  fout << " fv: " << camera_intrinsics[(int)INTRINSICS::fv] << "\n";
  fout << " u0: " << camera_intrinsics[(int)INTRINSICS::u0] << "\n";
  fout << " v0: " << camera_intrinsics[(int)INTRINSICS::v0] << "\n";

  fout.close();
}

/**
 * @brief
 *
 * @param cfgfile
 * @param landmarks
 */
inline void writeMapConfig(const std::string& cfgfile, const landmark_map_t& landmarks) {
  std::ofstream fout(cfgfile);

  fout << "Landmarks:\n";
  for (auto& entry : landmarks) {
    fout << " - {";
    fout << " HexID: " << "0x" << std::setfill('0') << std::setw(4) << std::hex << entry.first << std::setfill(' ');
    fout << ", x: " << std::fixed << std::setw(8) << entry.second.pose.position[(int)POINT::X];
    fout << ", y: " << std::fixed << std::setw(8) << entry.second.pose.position[(int)POINT::Y];
    fout << ", z: " << std::fixed << std::setw(8) << entry.second.pose.position[(int)POINT::Z];
    fout << ", qx: " << std::fixed << std::setw(8) << entry.second.pose.orientation[(int)QUAT::X];
    fout << ", qy: " << std::fixed << std::setw(8) << entry.second.pose.orientation[(int)QUAT::Y];
    fout << ", qz: " << std::fixed << std::setw(8) << entry.second.pose.orientation[(int)QUAT::Z];
    fout << ", qw: " << std::fixed << std::setw(8) << entry.second.pose.orientation[(int)QUAT::W];
    fout << " }\n";
  }

  fout.close();
}

}  // namespace stargazer
