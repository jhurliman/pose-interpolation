cmake_minimum_required(VERSION 3.14)
project(se3_foxglove LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(FetchContent)
# https://github.com/foxglove/foxglove-sdk/releases?q=sdk%2F&expanded=true
FetchContent_Declare(
  foxglove
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  URL https://github.com/foxglove/foxglove-sdk/releases/download/sdk%2Fv0.8.1/foxglove-v0.8.1-cpp-x86_64-unknown-linux-gnu.zip
  URL_HASH
    SHA256=d7a403435f543a6c5a8fbe431a72924152fc7f791c8e8e3e6cce4b2fd82ae467)
FetchContent_MakeAvailable(foxglove)

add_executable(se3_foxglove main.cpp)

target_include_directories(
  se3_foxglove PRIVATE ${foxglove_SOURCE_DIR}/include
                       ${foxglove_SOURCE_DIR}/include/foxglove)

file(GLOB FOXGLOVE_SOURCES CONFIGURE_DEPENDS "${foxglove_SOURCE_DIR}/src/*.cpp"
     "${foxglove_SOURCE_DIR}/src/server/*.cpp")
target_sources(se3_foxglove PRIVATE ${FOXGLOVE_SOURCES})
target_link_libraries(se3_foxglove
                      PRIVATE ${foxglove_SOURCE_DIR}/lib/libfoxglove.a)
