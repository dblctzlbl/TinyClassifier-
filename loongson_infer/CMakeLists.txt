cmake_minimum_required(VERSION 3.10)
project(loongson_cls_bench)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(ncnn REQUIRED)

add_executable(loongson_cls_bench main.cpp)
target_link_libraries(loongson_cls_bench PRIVATE ncnn ${OpenCV_LIBS})
