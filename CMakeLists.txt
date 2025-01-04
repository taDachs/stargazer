cmake_minimum_required(VERSION 3.8)
project(stargazer)

# ROS 2 dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(OpenCV REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(Ceres REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(ament_cmake_gtest REQUIRED)

# Compiler settings
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

###########
## Build ##
###########
# Add include and library directories
include_directories(
    include/${PROJECT_NAME}
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${CERES_INCLUDE_DIR}
)

# Declare a C++ library
add_library(${PROJECT_NAME} SHARED
    src/CeresLocalizer.cpp
    src/DebugVisualizer.cpp
    src/LandmarkCalibrator.cpp
    src/LandmarkFinder.cpp
)
target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBRARIES}
    yaml-cpp
    Ceres::ceres
)
ament_target_dependencies(${PROJECT_NAME}
    rclcpp
    OpenCV
    yaml-cpp
    Eigen3
    Ceres
)

# Install headers and library
install(
  DIRECTORY include/
  DESTINATION include
)
install(
  TARGETS ${PROJECT_NAME}
  EXPORT ${PROJECT_NAME}Targets
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
)

###############
## Testing ##
###############
if(BUILD_TESTING)
  # ConfigHandler Test
  ament_add_gtest(test_config_handler test/test_ConfigHandler.cpp WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/test)
  if(TARGET test_config_handler)
    target_include_directories(test_config_handler PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
      $<INSTALL_INTERFACE:include>
    )
    target_link_libraries(test_config_handler
        ${PROJECT_NAME}
        yaml-cpp
    )
  endif()

  # Landmark Test
  ament_add_gtest(test_landmark test/test_landmark.cpp)
  if(TARGET test_landmark)
    target_include_directories(test_landmark PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
      $<INSTALL_INTERFACE:include>
    )
    target_link_libraries(test_landmark
        ${PROJECT_NAME}
    )
  endif()

  install(
    TARGETS test_config_handler test_landmark
    DESTINATION lib/${PROJECT_NAME}/test
  )
endif()

###############
## Package ##
###############
ament_export_targets(${PROJECT_NAME}Targets HAS_LIBRARY_TARGET)
ament_export_dependencies(
    rclcpp
    OpenCV
    yaml-cpp
    Eigen3
    Ceres
)
ament_package()
