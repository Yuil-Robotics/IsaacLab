cmake_minimum_required(VERSION 3.8)

project(dog_description)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY config launch meshes urdf
  DESTINATION share/${PROJECT_NAME}
)

install(
  PROGRAMS scripts/urdf_live_reloader.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME urdf_live_reloader
)

ament_package()
