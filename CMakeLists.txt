cmake_minimum_required(VERSION 3.25)

project(LearningVulkan)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Vulkan REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(Stb REQUIRED)
# find_package(tinyobjloader CONFIG REQUIRED)

add_subdirectory(LearningVulkan)

