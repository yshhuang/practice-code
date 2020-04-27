cmake_minimum_required(VERSION 3.13)
project(learning_opencv3)
find_package(OpenCV)
set(CMAKE_CXX_STANDARD 14)
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(learning_opencv3 main.cpp)
target_link_libraries( learning_opencv3 ${OpenCV_LIBS} )

