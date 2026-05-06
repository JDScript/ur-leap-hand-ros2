cmake_minimum_required(VERSION 3.8)
project(hand_eye_calibration)

find_package(ament_cmake REQUIRED)

install(PROGRAMS
  scripts/collector.py
  scripts/tf_publisher.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  config
  DESTINATION share/${PROJECT_NAME}
)

install(FILES resource/hand_eye_calibration
  DESTINATION share/ament_index/resource_index/packages
)

ament_package()
