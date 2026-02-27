# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 望着天空的眼睛 <a15234181830@163.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

cmake_minimum_required(VERSION 3.10)
project(loongson_cls_bench)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(ncnn REQUIRED)

add_executable(loongson_cls_bench main.cpp)
target_link_libraries(loongson_cls_bench PRIVATE ncnn ${OpenCV_LIBS})
